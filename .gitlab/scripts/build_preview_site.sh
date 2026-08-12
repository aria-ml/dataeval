#!/usr/bin/env bash
#
# Assemble the merge request preview site published to GitLab Pages.
#
# Collects whichever artifacts are present into public/ and writes a landing
# page linking to them. Both inputs are optional: the docs job only runs when
# docs or source files change, while coverage runs on every merge request.
#
#   output/docs/html/ -> public/docs/
#   htmlcov/          -> public/coverage/
#
set -euo pipefail

mkdir -p public
links=""

if [ -d output/docs/html ]; then
  mv output/docs/html public/docs
  links="${links}<li><a href=\"docs/\">Documentation</a><span>Sphinx build of docs/source</span></li>"
fi

if [ -d htmlcov ]; then
  mv htmlcov public/coverage
  links="${links}<li><a href=\"coverage/\">Coverage report</a><span>Line coverage for src/dataeval</span></li>"
fi

if [ -z "$links" ]; then
  echo "error: neither output/docs/html nor htmlcov was found" >&2
  exit 1
fi

# Title the page after the merge request so a tab full of previews stays
# tellable apart. The title is author-supplied, so escape it before it lands
# in the markup; it is unset outside merge request pipelines.
html_escape() {
  printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

mr_title=$(html_escape "${CI_MERGE_REQUEST_TITLE:-}")
page_title="${CI_PROJECT_NAME} preview"
mr_line="Merge request !${CI_MERGE_REQUEST_IID}"

if [ -n "$mr_title" ]; then
  page_title="${page_title} — ${mr_title}"
  mr_line="${mr_line} &middot; ${mr_title}"
fi

cat > public/index.html <<EOF
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${page_title}</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff;
    --fg: #1f2328;
    --muted: #656d76;
    --line: #d8dee4;
    --link: #0969da;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0d1117;
      --fg: #e6edf3;
      --muted: #9198a1;
      --line: #30363d;
      --link: #4493f8;
    }
  }
  body {
    margin: 0 auto;
    padding: 3rem 1.5rem;
    max-width: 40rem;
    background: var(--bg);
    color: var(--fg);
    font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
  p.sub { margin: 0 0 2rem; color: var(--muted); }
  ul { list-style: none; margin: 0; padding: 0; }
  li {
    padding: 1rem 0;
    border-top: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    gap: .25rem;
  }
  li:last-child { border-bottom: 1px solid var(--line); }
  a { color: var(--link); text-decoration: none; font-weight: 600; }
  a:hover { text-decoration: underline; }
  li span { color: var(--muted); font-size: .875rem; }
  footer { margin-top: 2rem; color: var(--muted); font-size: .875rem; }
</style>
</head>
<body>
<h1>${CI_PROJECT_NAME} preview</h1>
<p class="sub">${mr_line} &middot; commit ${CI_COMMIT_SHORT_SHA}</p>
<ul>
${links}
</ul>
<footer><a href="${CI_PIPELINE_URL}">Pipeline ${CI_PIPELINE_ID}</a></footer>
</body>
</html>
EOF
