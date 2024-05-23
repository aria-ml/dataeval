import sys
import threading

from .mst_utils import Edge, MSTNode

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Main <file>", file=sys.stderr)
        sys.exit(-1)
    print_lock = threading.Lock()
    initial_port = 12345
    timeout = 0
    nodes = set()
    edges = []
    with open(str(sys.argv[1])) as file:
        for line in file:
            edge = line.split()
            timeout += 2 * int(edge[3])
            edges.append(
                Edge(
                    src_node=int(edge[0]),
                    src_port=initial_port + int(edge[0]),
                    dest_node=int(edge[1]),
                    dest_port=initial_port + int(edge[1]),
                    weight=int(edge[2]),
                    delay=int(edge[3]),
                )
            )
            edges.append(
                Edge(
                    src_node=int(edge[1]),
                    src_port=initial_port + int(edge[1]),
                    dest_node=int(edge[0]),
                    dest_port=initial_port + int(edge[0]),
                    weight=int(edge[2]),
                    delay=int(edge[3]),
                )
            )
            nodes.add(int(edge[0]))
            nodes.add(int(edge[1]))

    threads = []
    for node in nodes:
        threads.append(
            MSTNode(
                uid=node,
                port=initial_port + node,
                edges=[edge for edge in edges if edge.src_node == node],
                timeout=timeout,
            )
        )

    for x in threads:
        x.start()
