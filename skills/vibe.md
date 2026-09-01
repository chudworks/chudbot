# Vibe coding manual

Work only inside `/workspace`. This is a React 19, TypeScript, Vite, and SCSS
site managed with Bun. Inspect the existing project before editing. Use
`bun add <registry-package>` for public npm packages and keep `bun.lock`.
Never add Git, URL, `file:`, or `link:` dependencies, `.npmrc`, `bunfig.toml`,
backend code, secrets, or a top-level `__vibe` path.

Run `bun run build` before finishing. The deployed clean build is performed by
Chudbot after you stop, so do not commit or deploy. Finish with a short factual
summary.

The global `vibe` object is typed by `src/vibe.d.ts`. Call
`await vibe.identity()` to get the signed-in viewer's public Discord display
identity, guild display name, and site name. Do not assume identity is
available before the promise resolves. Use browser-side public APIs only.

Client-side routing must tolerate a direct navigation or refresh: Chudbot
serves `index.html` for document requests that do not match a built file.
Build a polished, responsive site with accessible HTML, clear hierarchy,
deliberate typography, and useful empty/error states.
