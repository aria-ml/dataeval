/* Make the HTML repr of a DataFrame easier to read: dtypes folded into the header, numbers right-aligned,
   click a header to sort, long tables scroll under a sticky header and take a row filter. */
(function () {
  "use strict";

  var TALL = 14; // rows above which a table scrolls and gets a filter box
  var NUMERIC_DTYPE = /^(i|u|f)\d+$|^decimal/i;
  var NUMBER = /^[-+]?(\d[\d,]*\.?\d*|\.\d+)(e[-+]?\d+)?$|^[-+]?(nan|inf)$/i;

  function isNull(text) {
    return text === "null" || text === "NaN" || text === "None" || text === "";
  }

  function numeric(text) {
    if (/^[-+]?inf$/i.test(text)) return text[0] === "-" ? -Infinity : Infinity;
    return parseFloat(text.replace(/,/g, ""));
  }

  function columnKinds(table, dtypes) {
    var rows = table.tBodies[0].rows;
    var width = table.tHead.rows[0].cells.length;
    var kinds = [];
    for (var c = 0; c < width; c++) {
      var kind;
      if (dtypes && dtypes[c]) {
        kind = NUMERIC_DTYPE.test(dtypes[c]) ? "num" : "text";
      } else {
        kind = "num";
        for (var r = 0; r < rows.length && kind === "num"; r++) {
          var cell = rows[r].cells[c];
          var text = cell ? cell.textContent.trim() : "";
          if (!isNull(text) && !NUMBER.test(text)) kind = "text";
        }
      }
      kinds.push(kind);
    }
    return kinds;
  }

  function enhance(table) {
    if (table.dataset.dfDone) return;
    table.dataset.dfDone = "1";
    var thead = table.tHead;
    var body = table.tBodies[0];
    if (!thead || !body || !thead.rows.length) return;

    // Polars prints the dtypes as a second header row of <td>; pandas never does.
    var dtypes = null;
    if (thead.rows.length > 1 && Array.prototype.every.call(thead.rows[1].cells, function (c) { return c.tagName === "TD"; })) {
      dtypes = Array.prototype.map.call(thead.rows[1].cells, function (c) { return c.textContent.trim(); });
      thead.deleteRow(1);
    }

    var kinds = columnKinds(table, dtypes);
    var head = thead.rows[0];
    var rows = Array.prototype.slice.call(body.rows);
    rows.forEach(function (row, i) {
      row._dfIndex = i;
      Array.prototype.forEach.call(row.cells, function (cell, c) {
        var text = cell.textContent;
        if (dtypes && dtypes[c] === "str" && text.length >= 2 && text[0] === '"' && text[text.length - 1] === '"') {
          cell.textContent = text.slice(1, -1);
        } else if (isNull(text.trim()) && text.trim() !== "") {
          cell.classList.add("df-null");
        }
        if (kinds[c] === "num") cell.classList.add("df-num");
      });
    });

    var state = { col: -1, dir: 0 };
    var arrows = [];
    Array.prototype.forEach.call(head.cells, function (th, c) {
      if (th.tagName !== "TH") return;
      var name = th.textContent;
      th.textContent = "";
      var arrow = document.createElement("span");
      arrow.className = "df-arrow";
      th.appendChild(arrow);
      th.appendChild(document.createTextNode(name));
      if (dtypes && dtypes[c]) {
        var dtype = document.createElement("span");
        dtype.className = "df-dtype";
        dtype.textContent = dtypes[c];
        th.appendChild(dtype);
      }
      if (kinds[c] === "num") th.classList.add("df-num");
      th.classList.add("df-sortable");
      th.title = "Click to sort";
      arrows[c] = arrow;
      th.addEventListener("click", function () { sortBy(c); });
    });

    function sortBy(c) {
      state.dir = state.col === c ? (state.dir === 1 ? -1 : state.dir === -1 ? 0 : 1) : 1;
      state.col = c;
      arrows.forEach(function (a, i) { if (a) a.textContent = i === c ? (state.dir === 1 ? "▲" : state.dir === -1 ? "▼" : "") : ""; });
      var sorted = rows.slice();
      if (state.dir !== 0) {
        sorted.sort(function (a, b) {
          var x = a.cells[c].textContent.trim();
          var y = b.cells[c].textContent.trim();
          // A real null or NaN carries df-null; a string that reads "null" does not.
          var missingX = a.cells[c].classList.contains("df-null");
          var missingY = b.cells[c].classList.contains("df-null");
          if (missingX || missingY) return missingX === missingY ? a._dfIndex - b._dfIndex : missingX ? 1 : -1;
          var order = kinds[c] === "num" ? numeric(x) - numeric(y) : x.localeCompare(y, undefined, { numeric: true });
          return order === 0 ? a._dfIndex - b._dfIndex : order * state.dir;
        });
      }
      sorted.forEach(function (row) { body.appendChild(row); });
    }

    var scroll = document.createElement("div");
    scroll.className = "df-scroll" + (rows.length > TALL ? " df-tall" : "");
    table.parentNode.insertBefore(scroll, table);
    scroll.appendChild(table);

    if (rows.length > TALL) {
      var filter = document.createElement("input");
      filter.type = "search";
      filter.className = "df-filter";
      filter.placeholder = "Filter " + rows.length + " rows";
      filter.setAttribute("aria-label", "Filter rows");
      filter.addEventListener("input", function () {
        var term = filter.value.trim().toLowerCase();
        rows.forEach(function (row) { row.hidden = term !== "" && row.textContent.toLowerCase().indexOf(term) === -1; });
      });
      scroll.parentNode.insertBefore(filter, scroll);
    }
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll("table.dataframe"), enhance);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  // With navigation.instant the theme swaps page content without a reload and announces it on document$.
  if (window.document$ && window.document$.subscribe) {
    window.document$.subscribe(init);
  }
})();
