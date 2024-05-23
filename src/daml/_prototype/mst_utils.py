import json
import socket
import sys
import threading
import time
from _thread import start_new_thread

# edge states
BASIC = "BASIC"
BRANCH = "BRANCH"
REJECTED = "REJECTED"

# node states
SLEEP = "SLEEP"
FOUND = "FOUND"
SEARCH = "SEARCH"


class Message:
    def __init__(self, uid, msg_type, value):
        self.uid = uid
        self.type = msg_type
        self.value = value


class Edge:
    def __init__(self, src_node, src_port, dest_node, dest_port, weight, delay):
        self.src_node = src_node
        self.src_port = src_port
        self.dest_node = dest_node
        self.dest_port = dest_port
        self.weight = weight
        self.delay = delay
        self.state = BASIC


class ConnectRequests:
    def __init__(self):
        self.requests = {}

    def insert(self, p, level):
        if p not in self.requests:
            self.requests[p] = level

    def get_least_level_req(self):
        if self.requests:
            return min(self.requests, key=self.requests.get)  # type: ignore

    def least_level(self):
        if self.requests:
            return self.requests[self.get_least_level_req()]
        else:
            return sys.maxsize


class MSTNode(threading.Thread):
    def __init__(self, uid, port, edges, timeout):
        threading.Thread.__init__(self)
        self.uid = int(uid)
        self.port = int(port)
        self.edges = edges
        self.timeout = timeout
        self.state = SLEEP  # current state of node, initially sleep
        self.level = 0  # current level of node
        self.core = None  # determines the fragment
        self.active = True  # is MST still running or ended
        self.best_weight = sys.maxsize  # weight of best edge found
        self.best_path = []  # best path found so far
        self.test_edge = None  # edge that test message has sent to
        self.connect_requests = ConnectRequests()  # store connect requests received
        self.find_count = None  # search request sent
        self.find_source = None  # node that get search state from
        self.test_requests = set()  # store test requests received
        self.sent_connect_to = None  # node that connect message has sent to
        # true if has sent initiate message and waiting for report
        self.expect_core_report = None
        self.report_over = False  # true if has sent report message in response of initiate
        self.other_core_node = None  # set to other core node in merge process
        self.discovery = None  # mode looking for new nodes to connect
        self.test_over = None  # true if no one is remaining to send test message
        self.leader = -1  # to :))

    # starting function of thread initializing program
    def run(self):
        start_new_thread(self.WakeUp, ())  # start node itself
        start_new_thread(self.main, ())  # initialize main loop
        # end program after timeout
        start_new_thread(self.do_end, (self.timeout,))
        self.node_listener()  # start a listen process for node port to receive messages

    # simulates message send with channels having delay
    def node_sender(self, message, edge):
        time.sleep(int(edge.delay))  # simulates channel delay
        try:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.connect(("127.0.0.1", edge.dest_port))
            connection.sendall(json.dumps(message.__dict__).encode("utf-8"))
        except:  # happens if dest node listener has stopped  # noqa: E722
            pass

    # create message and pass it to message channel
    def send_message(self, msg_type, value, edge):
        start_new_thread(  # create instance of message transfer
            self.node_sender,
            (Message(self.uid, msg_type, value), edge),
        )

    # get outgoing edge to node with specific id
    def getEdgeById(self, edge_id):
        return next(x for x in self.edges if int(x.dest_node) == int(edge_id))

    # changes node state and sends connect message to minimum weight outgoing edge
    def WakeUp(self):
        if self.state == SLEEP:
            self.state = FOUND
            min_weight_outgoing_edge = min(self.edges, key=lambda x: x.weight)  # type: ignore
            self.send_message("Connect", str(self.core) + "," + str(self.level), min_weight_outgoing_edge)
            min_weight_outgoing_edge.state = BRANCH
            self.sent_connect_to = min_weight_outgoing_edge.dest_node

    # receive messages and call intended function
    def node_listener(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", self.port))
        s.listen(1)
        s.settimeout(5)
        while self.active:
            try:
                conn, addr = s.accept()
                data = conn.recv(1024)
                m = Message(**json.loads(data, encoding="utf-8"))
                print(
                    '{"from": '
                    + str(m.uid)
                    + ', "to": '
                    + str(self.uid)
                    + ', "type": "'
                    + str(m.type)
                    + '", "value": '
                    + str(m.value)
                    + "}"
                )  # +str(self.uid))
                if m.type == "End":
                    start_new_thread(self.do_end, (0,))
                elif m.type == "Connect":
                    data = str(m.value).split(",")
                    start_new_thread(self.do_connect, (m.uid, data[0], int(data[1])))  # Core, Level
                elif m.type == "Test":
                    data = str(m.value).split(",")
                    start_new_thread(self.do_test, (m.uid, int(data[0]), int(data[1])))  # Core, Level
                elif m.type == "Initiate":
                    data = str(m.value).split(",")
                    start_new_thread(
                        self.do_initiate,
                        (
                            m.uid,
                            int(  # Core, Level, State, merge
                                data[0]
                            ),
                            int(data[1]),
                            data[2],
                            bool(data[3]),
                        ),
                    )
                elif m.type == "Reject":
                    start_new_thread(self.do_reject, (m.uid,))
                elif m.type == "Report":
                    data = str(m.value).split(",")
                    if data[1] != "" and data[1]:
                        data[1] = data[1].split("::")[0]  # TODO: only took first element of split - is this correct?
                    start_new_thread(self.do_report, (m.uid, int(data[0]), data[1]))
                elif m.type == "ChangeRoot":
                    data = str(m.value).split(",")
                    if data[0] != "" and data[0]:
                        data[0] = data[0].split("::")[0]  # TODO: only took first element of split - is this correct?
                    start_new_thread(self.do_change_root, (m.uid, data[0]))
                elif m.type == "Accept":
                    start_new_thread(self.do_accept, (m.uid,))
            except:  # noqa: E722
                pass
            time.sleep(0.1)
        s.close()

    # called upon receiving Connect message
    # inserts message to connect requests received list
    def do_connect(self, Sender, Core, Level):
        self.WakeUp()
        self.connect_requests.insert(Sender, Level)

    # called upon receiving Test message
    # inserts message to test requests received list
    def do_test(self, Sender, Core, Level):
        self.WakeUp()
        self.test_requests.update({(Level, Core, Sender)})

    # called upon receiving End message
    # prints branches
    def do_end(self, timeout):
        time.sleep(timeout)
        self.active = False
        branches = ",".join(str(edge.dest_node) for edge in self.edges if edge.state == BRANCH)
        print(str(self.uid) + " is connected to: " + branches)

    # called upon receiving Initiate message
    # may happen also on merges, send initiate message to all branches
    def do_initiate(self, Sender, Core, Level, State, merge):
        self.level = Level
        self.core = Core
        self.state = State
        self.best_weight = sys.maxsize
        self.best_path = []

        if merge is True:
            self.other_core_node = Sender
            self.leader = max(Sender, self.leader)
        else:
            self.other_core_node = None

        self.getEdgeById(Sender).state = BRANCH

        my_branches = {edge for edge in self.edges if edge.state == BRANCH} - {self.getEdgeById(Sender)}

        self.find_count = 0
        for b in my_branches:
            self.send_message(
                "Initiate", str(self.core) + "," + str(self.level) + "," + str(self.state) + "," + str(False), b
            )
            if self.state == SEARCH:
                self.find_count += 1

        if self.state == SEARCH:
            self.find_source = Sender
            self.report_over = False
            self.test()

    # called upon receiving Reject message
    # happens in reply to Test
    def do_reject(self, Sender):
        if self.getEdgeById(Sender).state == BASIC:
            self.getEdgeById(Sender).state = REJECTED

        self.test()

    # called upon receiving Accept message
    # happens in reply to Test
    def do_accept(self, Sender):
        if self.getEdgeById(Sender).weight < self.best_weight:
            self.best_path = [self.uid, Sender]
            self.best_weight = self.getEdgeById(Sender).weight
        self.test_over = True

    # called upon receiving Report message
    # happens in reply to initiate
    def do_report(self, Sender, weight, path):
        if Sender == self.other_core_node:
            self.expect_core_report = False
        else:
            if self.find_count and self.find_count > 0:
                self.find_count -= 1
        if weight < self.best_weight:
            self.best_path = [self.uid] + path
            self.best_weight = weight

    # called upon receiving ChangeRoot message
    # happens for connecting 2 disjoint fragments
    def do_change_root(self, Sender, path):
        if len(path) == 0:
            node = Sender
            self.send_message("Connect", str(self.core) + "," + str(self.level), self.getEdgeById(node))
            self.getEdgeById(node).state = BRANCH
            self.sent_connect_to = node

        elif len(path) > 1:
            hd = path[1]
            tl = path[2:]
            self.send_message("ChangeRoot", str("::".join(str(v) for v in tl)), self.getEdgeById(hd))

    # happens if no one is remaining to send test message and got response from all search requests sent
    def report(self):
        self.test_over = None
        self.state = FOUND
        self.send_message(
            "Report",
            str(self.best_weight) + "," + str("::".join(str(node) for node in self.best_path)),
            self.getEdgeById(self.find_source),
        )
        self.report_over = True

    # happens if received all reports and don't have to report
    def fragment_connect(self):
        self.discovery = True
        self.expect_core_report = None
        self.report_over = False

        if self.best_path[1] != self.other_core_node:
            hd = self.best_path[1]
            tl = self.best_path[2:]
            self.send_message("ChangeRoot", str("::".join(str(v) for v in tl)), self.getEdgeById(hd))

    # happens if received connect from someone whom node sent connect before
    def merge(self, node):
        connection_edge = self.getEdgeById(node)

        new_core = connection_edge.weight  # weight of edge of 2 fragment

        connection_edge.state = BRANCH
        self.send_message(
            "Initiate", str(new_core) + "," + str(self.level + 1) + "," + str(SEARCH) + "," + str(True), connection_edge
        )  # initiate with merge=True
        self.expect_core_report = True
        self.discovery = False

    # happens if received connect from someone with lower level
    def absorb_node(self, node):
        self.send_message(
            "Initiate",
            str(self.core) + "," + str(self.level) + "," + str(self.state) + "," + str(False),
            self.getEdgeById(node),
        )
        self.getEdgeById(node).state = BRANCH

        if self.state == SEARCH:
            if self.find_count:
                self.find_count = self.find_count + 1
            else:
                self.find_count = 1

    # send test messages
    # happens on initiate and upon receiving reject
    def test(self):
        basic_edges = [edge for edge in self.edges if edge.state == BASIC]
        self.test_over = False

        if basic_edges:
            self.test_edge = min(basic_edges, key=lambda edge: edge.weight)
            self.send_message("Test", str(self.core) + "," + str(self.level), self.test_edge)
        else:
            self.test_over = True

    def process_test_requests(self):
        to_remove = set()

        for L, F, j in self.test_requests:
            # if same fragment -> reject message is sent back
            if self.core == F:
                self.send_message("Reject", "", self.getEdgeById(j))
                to_remove.update({(L, F, j)})

            # if different fragment and level in message < self.level -> an accept message is sent back.
            elif self.level >= L:
                self.send_message("Accept", "", self.getEdgeById(j))
                to_remove.update({(L, F, j)})

        self.test_requests -= to_remove

    def main(self):
        while self.active:
            if self.sent_connect_to in set(self.connect_requests.requests):  # merge
                self.merge(self.sent_connect_to)
                self.connect_requests.requests.pop(self.sent_connect_to)
                self.sent_connect_to = None

            elif self.connect_requests.least_level() < self.level:  # absorb
                node = self.connect_requests.get_least_level_req()
                self.absorb_node(node)
                self.connect_requests.requests.pop(node)

            # if different fragment and level in message < self.level –> no reply is sent until situation changes.
            elif bool(
                {True for (L, F, j) in self.test_requests if self.core == F or (self.core != F and self.level >= L)}
            ):
                self.process_test_requests()

            elif self.test_over and self.find_count == 0:
                self.report()

            elif self.expect_core_report is False:
                if self.report_over is True and self.discovery is False:
                    self.fragment_connect()

            time.sleep(0.1)
