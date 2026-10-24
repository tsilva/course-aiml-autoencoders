# Colab browser workflow

Load the bundled Browser skill/runtime and initialize its documented
`browser-client` through the persistent Node REPL. Select
`agent.browsers.get("iab")`. Discover the installed runtime instead of hard-coding
a plugin cache version. Use documented Browser APIs and visible UI state.

If a Colab tab already exists, keep the user's notebook intact and use a new tab
or newly uploaded validation copy. An uploaded harness creates a Drive notebook
in the connected Colab account; do not share it or modify sharing permissions.

## Upload and run

1. Open `https://colab.research.google.com/` in the in-app browser.
2. Use **File → Upload notebook → Browse** and the browser file-chooser API to
   select the helper's generated `.ipynb` from its absolute local path.
3. Connect to a fresh default CPU runtime. If this copy already has state, use
   **Runtime → Disconnect and delete runtime**, then connect again. Avoid
   resetting a runtime belonging to the user's existing work.
4. Click **Run all** and confirm execution if Colab asks about uploaded code.
   Read current UI labels/locators; the UI can change. Never infer completion from
   the Run-all button or elapsed time alone.
5. Inspect the terminal per-lesson lines and final JSON report. Captured outputs
   include all lesson displays for semantic review; output-bearing copies and a
   JSON report also live in the printed runtime directory under `runs/`.

If the output is in an iframe, inspect its accessible frame/DOM snapshot rather
than assuming the output container's empty `innerText` means nothing ran. Use
output fullscreen or screenshots when image review requires it. Do not change
the viewport just to make evidence look better.

## Troubleshooting and evidence

- Close Colab's release-notes pane if it obstructs controls. Refresh DOM or
  accessibility state after dialogs/navigation; stale node indices are unreliable.
- Prefer file upload over pasting a large encoded payload into an editor. For
  small diagnostic cells, focus the visible editor, replace its content through
  documented UI input, and verify what was entered before running it.
- Read failures before reconnecting or rerunning. A quota, authentication, trust,
  or allocation block leaves validation incomplete. Never use local execution
  as the final substitute for actual Colab.
- Save the Colab URL, visible report, screenshots of selected meaningful plots,
  and environment versions beside the local manifest. Preserve an accessible
  deliverable tab when supported. Do not expose account email addresses or tokens.
