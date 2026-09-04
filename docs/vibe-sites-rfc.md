# RFC: Vibe — websites from Discord

Status: accepted for implementation.

Date: 2026-09-01.

## Note for the implementer

This document is the handoff for Vibe version 1. Vibe is a toy for 10 to 20
friends across a few Discord servers. Build the simplest thing that matches
this document. Follow `AGENTS.md` and `docs/rust-style.md`, implement Phases 1
through 4, and keep the workspace compiling with tests passing after each
phase. Add config examples, `check-config` diagnostics, migrations, and deploy
notes in the same change as the behavior they support.

You may change the repository, run local builds, and use mocked tests and
throwaway local containers. You may not run `serve.sh deploy`, migrate the
production database, change the host, Cloudflare, or the Discord application,
or call live Discord or model providers from tests. Write a short runbook for
those steps instead.

When this document and the repository disagree, keep the safer behavior and
report the conflict rather than silently changing the design.

## Summary

Vibe lets a guild member ask Chudbot for a website and get a link back a few
minutes later:

> @Chudbot make a vibe site that plots mortgage amortization curves, call it
> "chud-mortgages"

```text
Site:   https://chud-mortgages.vibe.example/
Source: https://src.vibe.example/sites/chud-mortgages
```

A coding agent writes a React site inside a throwaway Docker container.
Chudbot builds it from a clean checkout, commits it to a local Git repository,
and serves the result from the existing `chudbot` process.

Sites are `🔒 protected` by default: viewers log in with Discord and Chudbot
checks that they remain members of the owning server. An owner or administrator
can make the deployed site public through `vibe_manage`; its source and history
remain protected. Sites are static. There is no custom server code. The one
backend feature in version 1 is `vibe.identity()`, a small JavaScript call that
tells a protected site who is looking at it and returns `null` on a public site.

Everything else stays deliberately small:

- one template (React, TypeScript, Vite, SCSS), one package manager (Bun), one
  Docker image;
- coding happens only through Chudbot in Discord; there is no CLI, upload, or
  Git remote;
- the coding agent gets three tools: `read`, `edit`, and `shell`;
- one Cloudflare Tunnel and one wildcard DNS record serve every site;
- Postgres, local bare Git repositories, and a local artifact directory hold
  all state.

## Goals

- Create or change a site from an ordinary Discord message.
- Let Chudbot pick a good available name when the user does not give one.
- Let "change that site to add dark mode" work later, even in a new
  conversation.
- Make the URL shareable with the guild right away.
- Keep Git history and make it browsable.
- Keep the whole thing inside the existing single-host Chudbot deployment.
- Keep model mistakes recoverable: nothing is published half-built or
  overwritten silently.

## Non-goals for version 1

- Backend code, cron jobs, databases, or file uploads for sites. Sites may
  call public APIs from the browser.
- A CLI, local checkout, or Git push.
- Multiple templates or build stacks.
- Public sites, custom domains, or cross-guild sharing.
- Importing Discord attachments or Chudbot media into a site.
- Private source. Anyone who can view a site can view its source.
- Protecting viewers from a guild member who deliberately writes hostile
  JavaScript. The guild is a group of friends. Chudbot protects its own host,
  secrets, and sessions, not viewers from each other.
- Adding Discord login to the existing trace viewer.

## Terms

- **Site**: a name, a guild, an owner, optional editors, and a status.
- **Revision**: one Git commit plus the built files for it. Every successful
  job creates one.
- **Active revision**: the revision the site's hostname serves.
- **Job**: one coding-agent run, with its container and checkout.
- **Workspace**: the throwaway checkout mounted into the job's container.
- **Sandbox image**: the one Docker image every job uses.
- **Skill**: a Markdown file of instructions attached to an agent through
  config.

## How it works

1. The conversation agent sees a Vibe request and calls the `vibe` tool with
   `create` or `edit`, a site name, and the task.
2. Chudbot builds the actor (user, guild, conversation, turn) from the Discord
   message. The model cannot set these.
3. For `create`, Chudbot inserts the site row. A unique constraint on the name
   settles any race before a model or container starts. For `edit`, it checks
   that the actor may edit the site and that no other job is running on it.
4. Chudbot checks out the site's current commit (or the template for a new
   site) into a workspace and starts a container from the sandbox image.
5. The coding agent works with `read`, `edit`, and `shell` until it answers
   with a summary. It has no commit or deploy tool.
6. Chudbot copies the source out of the workspace, validates it, and builds it
   in a fresh container. A failed build goes back to the same agent as a
   repair turn, at most twice.
7. Chudbot commits the source to the site's bare repository, stores the built
   files, marks the revision active in one Postgres transaction, and deletes
   the workspace and containers.
8. The conversation agent replies with the site and source links.

If anything fails after the site row exists, the previous active revision
stays live. A brand-new site whose first job fails is deleted so the name is
free again.

Jobs can take several minutes, so the job posts a short status message to the
channel when coding starts, when the clean build starts, and on each repair
turn, using the existing status-message support. Jobs are limited to 15
minutes.

### Names

When the user gives a name, Chudbot uses it. If it is taken, Chudbot says so
and offers alternatives; it never silently substitutes. When the user gives no
name, the conversation agent invents a few short descriptive candidates,
checks them with `vibe_check_names`, and picks the best available one without
asking. Availability is a hint: the insert in step 3 is what claims the name,
and a lost race returns `name_unavailable` so the agent can try another
candidate.

### Finding an existing site

For "change that site", the conversation agent calls `vibe_list_sites`. Sites
touched in the current conversation come first, then the actor's own sites by
most recent change, then the rest of the guild's sites. Each result has the
name, description, the actor's role, and links. If two results are plausible,
the agent asks rather than guesses.

`create` and `edit` are separate. A create on an existing name fails; it never
becomes an edit.

## Who can do what

| Action | Guild member | Editor | Owner | Chudbot admin |
| --- | --- | --- | --- | --- |
| View site and source | yes | yes | yes | yes |
| Edit, roll back | no | yes | yes | yes |
| Add or remove editors, archive, restore | no | no | yes | yes |
| Purge | no | no | no | yes |

- Viewing always requires current membership in the site's guild, admins
  included.
- Sites can only be created from a guild channel, never a DM.
- The owner is the author of the Discord message. A tool argument cannot
  change that.
- Adding an editor checks that they are in the guild. Editors and owners who
  leave the guild lose access until they return.
- Archive hides the site and keeps its name and history. Restore brings it
  back. Purge deletes everything and is an operator command
  (`chudbot vibe purge <name>`), not a chat action.
- Ownership transfer is not in version 1.

Two config switches limit the rollout. `[vibe].enabled` turns the feature off
entirely. `[vibe.access]` can restrict Vibe to listed guilds and, with
`admins_only`, restrict creating and editing to the existing `[bot].admins`
list. Guild members can still view existing sites when `admins_only` is set. A
guild removed from the allowlist keeps its data, but its sites stop serving
until it is allowed again.

These checks run in Rust before any model or container starts and on every
HTTP request. Hiding the tools from the model in an ineligible guild is a
courtesy; the Rust check is the enforcement.

## Architecture

```text
Discord message
      |
      v
conversation agent ---- vibe_check_names, vibe_list_sites, vibe_manage
      |
      | vibe tool (create / edit)
      v
Vibe job runner
      |-- workspace checkout on disk
      |-- coding container (read / edit / shell)
      |-- clean-build container
      |-- bare Git repo  +  artifact directory  +  Postgres
      v
reply with links

Browser --> Cloudflare (TLS, tunnel) --> chudbot on 127.0.0.1
      vibe.example        explainer, login, logout, OAuth callback
      src.vibe.example    read-only source browser
      <name>.vibe.example session -> membership -> static files + /__vibe/ API
```

Postgres holds sites, revisions, jobs, and sessions. The bare repositories
hold source history. The artifact directory holds built files. Postgres
decides which commit is current.

### Crate changes

- `chudbot-api`: a `vibe` module with ids, the `VibeStorage` trait, the
  `VibeIdentityProvider` trait (login and guild-membership checks), and the
  `VibeActor` type. No Axum, SQLx, or Twilight types.
- `chudbot-vibe` (new): names, access checks, Git and artifact storage, the
  sandbox and job runner, the three coding tools, and the `vibe.js` API.
  Generic over the two traits above with static dispatch.
- `chudbot-storage-sqlx`: implements `VibeStorage` and embeds the migrations.
- `chudbot-discord`: implements `VibeIdentityProvider` with the Discord OAuth
  code flow and the bot's "Get Guild Member" endpoint. OAuth tokens never
  leave this crate and are dropped after login.
- `chudbot-bot`: skills, the `vibe_coder` subagent policy, and the
  conversation tools.
- `chudbot-web`: host-based routing in front of the existing trace-viewer
  routes, site serving, login, and the source browser.
- `chudbot-bin`: config, startup checks, the `vibe purge` command, and running
  the job runner alongside the bot and web server under the existing
  supervisor.

## Hosts and names

Version 1 uses the dedicated domain `vibe.example`. The existing
`chudbot.example.com` trace viewer is untouched; requests for other hosts never
reach Vibe routes and Vibe hosts never fall through to the trace viewer.

- `vibe.example`: a short explainer page, `/login`, `/logout`, and
  `/oauth/callback`.
- `src.vibe.example`: the source browser.
- `<name>.vibe.example`: one user site each.

Reserved labels: `src`, `www`, `api`, `admin`, `static`, `status`, `mail`,
and anything in `[vibe].reserved_names`.

A site name is one lowercase DNS label, 3 to 63 characters, matching:

```text
^[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])$
```

Names starting with `xn--` are rejected. Names are global because they share
one DNS namespace.

The host router lowercases the host, strips a port, and rejects anything that
is not the apex, a reserved label, or exactly one label under the base domain.
An unknown site is a 404 without a login redirect. The site is always derived
from the host, never from a query parameter.

`/__vibe/` is reserved on every site host for the SDK and API. Source may not
contain a top-level `__vibe` entry.

## Login and sessions

Vibe uses the Discord authorization-code flow with the `identify` scope only.
A browser that opens a protected site without a session is redirected to
`https://vibe.example/login?return=<url>`. Asset and API requests get a 401
instead, so a stylesheet never receives a login page. Only `return` URLs under
`vibe.example` are accepted. The OAuth `state` value is stored server-side,
expires after 10 minutes, and is single-use.

After the callback, Chudbot creates a session row and sets one cookie:

```text
vibe_session=<random>; Domain=.vibe.example; Secure; HttpOnly; SameSite=Lax; Path=/
```

Only a hash of the cookie value is stored. Sessions last for the configured
`[vibe.auth].session_days`. `/logout` revokes the session and clears the cookie.

As an OAuth-free alternative, a current member of an allowed server can ask
Chudbot to DM a login link to themselves or another current member. Chudbot
checks the target's current membership before creating the link. The raw token
is delivered only to the target's DM; the requester-facing tool result and
stored trace contain delivery metadata but never the token or URL. This makes
requesting a link for someone else safe without granting the requester that
person's session.

Direct-login links are bearer credentials, expire after 10 minutes, and are
single-use. Only their SHA-256 hashes are stored. `GET /login/direct?token=...`
atomically consumes the link, creates a fresh browser session for the bound
platform user, sets the normal `vibe_session` cookie, and redirects to the apex
host. The resulting session lifetime is `[vibe.auth].session_days`, exactly as
for OAuth-created sessions. Failed DM delivery consumes the new link so an
undelivered credential cannot remain active. Discord embeds are suppressed on
the DM to avoid link-preview redemption.

One cookie for the whole domain is a deliberate simplification. A hostile site
under `vibe.example` could set a sibling's cookie, which among friends is a prank
at worst. If that ever matters, switch to per-host cookies with a login
handoff.

On every protected site or API request, and every source-browser request,
Chudbot checks that the session's user is a current member of the site's guild
with the bot's "Get Guild Member" endpoint. Positive results are cached in
memory for 5 minutes, negative results for 30 seconds. Public deployed-site
requests skip this check. If Discord is unreachable and nothing is cached, a
protected request fails with a 503 and a job will not start.

The OAuth callback URL must be registered on the Discord application. That is
a runbook step.

## Data

### Tables

| Table | Columns |
| --- | --- |
| `vibe_sites` | `id`, `name` (unique), `platform`, `guild_id`, `owner_user_id`, `description`, `status` (`creating`, `active`, `archived`), `access` (`protected`, `public`; defaults to `protected`), `active_revision_id`, `running_job_id`, timestamps |
| `vibe_site_editors` | `site_id`, `platform`, `user_id`, `added_by_user_id`, `added_at`; primary key `(site_id, platform, user_id)` |
| `vibe_revisions` | `id`, `site_id`, `ordinal`, `parent_revision_id`, `commit_oid`, `image_id`, `message`, `build_log` (bounded), `actor_user_id`, `conversation_id`, `turn_id`, `job_id`, `created_at`; unique `(site_id, ordinal)` |
| `vibe_jobs` | `id`, `site_id` (nullable), `site_name`, `action`, `actor_user_id`, `platform`, `guild_id`, `conversation_id`, `turn_id`, `tool_use_id` (unique), `state`, `error`, timestamps |
| `vibe_sessions` | `token_hash` (primary key), `platform`, `user_id`, `created_at`, `expires_at`, `revoked_at` |
| `vibe_oauth_states` | `state_hash` (primary key), `return_url`, `expires_at`, `consumed_at` |
| `vibe_login_links` | `token_hash` (primary key), `platform`, `guild_id`, `user_id`, `requested_by_user_id`, `expires_at`, `consumed_at` |
| `vibe_collection_documents` | `site_id`, `collection`, `id`, JSONB `document`, `inserted_by`, `inserted_at`, `updated_by`, `updated_at`; primary key `(site_id, collection, id)` |

Revisions are never updated or deleted except by purge. `running_job_id` is
the per-site lock: a site with a running job rejects a second one. The unique
`tool_use_id` makes a retried tool call return the existing job instead of
starting another.

The site description is the first line of the task that created it,
truncated. It is search text for `vibe_list_sites`, not instructions.

The existing trace machinery already records nested subagent transcripts
inside the parent tool call, so the coding agent's full history shows up in
the normal trace viewer. No extra trace tables are needed.

### Disk

```text
$CHUDBOT_DIR/vibe/
  repos/<site-id>.git/               bare repositories, no remotes
  workspaces/<job-id>/               throwaway checkouts, deleted after the job
  artifacts/<site-id>/<revision-id>/ built dist/ trees, one per revision
```

`serve.sh deploy` treats this directory as persistent data and never replaces
it. Back up Postgres and this directory together. Old revisions are kept;
purge removes a site's repository and artifacts.

### Git

Each site has one bare repository with one branch, `main`. Chudbot runs the
`git` binary on the host with fixed plumbing commands, empty global config,
hooks disabled, and `--` before paths. The workspace's own `.git` directory is
throwaway; the agent may do whatever it likes to it.

At the end of a job the host reads the workspace tree itself. It skips
`.git`, `node_modules`, and `dist`, and rejects symlinks, hard links, device
files, nested `.git` or `.gitmodules`, `__vibe`, and paths with `..`, empty
segments, backslashes, or NULs. It then writes the blobs and tree to the bare
repository, creates the commit, writes the Postgres rows, and finally moves
`main`. Postgres is authoritative: if the process dies between the commit and
the row, the orphan commit is harmless and `main` is repaired at startup.
Every job starts from the commit Postgres names, not from `main`.

The commit author is a no-reply identity derived from the Discord user id.
The commit message is `Create <name>` or `Update <name>`, a blank line, the
task, and the agent's final summary, each truncated to a sensible length.

### Limits

Defaults, all configurable:

| Limit | Default |
| --- | --- |
| Source files per site | 2,000 |
| Single file | 10 MiB |
| Source tree, and built output | 50 MiB each |
| Running jobs per guild | 2 |
| Running jobs per site | 1 |
| Job wall clock | 15 minutes |
| One shell command | 60 seconds |
| Clean build, including install | 5 minutes |
| Container memory / CPUs / PIDs | 1 GiB / 2 / 256 |

## The build contract

One template and one image, versioned with the repository under
`vibe-sandbox/`:

```text
vibe-sandbox/
  Dockerfile
  template/
    package.json, bun.lock, index.html, vite.config.ts, tsconfig.json
    src/main.tsx, src/App.tsx, src/styles.scss
```

The Dockerfile pins Bun and Node.js and installs Git and `rg`. `serve.sh
deploy` builds it locally as `chudbot-vibe-sandbox:latest` and never pushes
it. Each revision records the image id it was built with.

The build is:

```text
bun install --frozen-lockfile
bun run build
```

and must produce `dist/index.html`. A site may add any public package from
`registry.npmjs.org` with `bun add`; `bun.lock` is committed with the source.
Git, URL, `file:`, and `link:` dependencies and repository `.npmrc` or
`bunfig.toml` files are rejected at export. Bun's `trustedDependencies` works
as usual.

After the clean build, Chudbot checks the output size and inserts the SDK tag
into `dist/index.html`:

```html
<script defer src="/__vibe/sdk/v1/vibe.js"></script>
```

Changing the template or image affects new sites and the next edit of
existing ones. Already-built revisions keep serving.

## The sandbox

Every job uses two containers from the sandbox image: the coding container
the agent works in, and a fresh clean-build container that installs
dependencies and builds the exported source. The clean build is what gets
deployed; the agent's own build output is ignored.

Containers run:

- as a non-root user, with `--cap-drop=ALL`, `no-new-privileges`, a read-only
  root filesystem, and tmpfs scratch;
- with memory, CPU, PID, and wall-clock limits;
- with only the workspace mounted and an explicit environment allowlist;
- with no Docker socket, Chudbot config, secrets, bare repositories,
  artifacts, or other workspaces;
- with outbound internet so `bun install` works.

The host firewall drops traffic from the Docker bridge to private address
ranges so a bad npm package cannot poke at the home network. That rule is a
runbook step and the entire network policy.

The model never sees a container id, image name, or host path. Shell commands
run as `/bin/bash -lc` inside the job's container. Docker arguments are built
from typed values, never from model text. Containers are force-removed when
the job ends, times out, is cancelled, or Chudbot shuts down. If Docker or the
image is unavailable, new jobs fail and existing sites keep serving.

Chudbot talks to the host's Docker daemon at the configured socket. Rootless
Docker is a good idea but not required.

## Agents

### Skills

A skill is a Markdown file attached to an agent by config:

```toml
[bot.skills.vibe]
path = "skills/vibe.md"

[bot.skills.vibe_conversation]
path = "skills/vibe-conversation.md"

[bot.agents.default]
skills = ["vibe_conversation"]

[bot.agents.vibe_coder]
skills = ["vibe"]
```

`check-config` requires each path to be a UTF-8 file under 64 KiB and rejects
unknown skill names. Files load at startup; changing one takes a restart.
Skill text becomes an ordinary labeled instruction part, so the existing
prompt snapshotting applies. Skills grant no tools or permissions, and nothing
inside a site repository is ever treated as a skill.

`skills/vibe.md` is the coding manual: the project layout, the Bun commands,
npm rules, `vibe.identity()`, durable collections and ephemeral watches/rooms,
SPA routing, design expectations, and "stop after a summary; Chudbot commits
and deploys". `skills/vibe-conversation.md` is the workflow: create versus
edit, inventing and checking names, finding sites, and how to report links.

### Configuration

```toml
[vibe]
enabled = true
base_domain = "vibe.example"
root_dir = "vibe"

[bot.agents.default.subagents.vibe]
agent = "vibe_coder"
tool_policy = "vibe_coder"
description = "Create or change a Vibe website and deploy it."

[bot.agents.vibe_coder]
provider = "grok"
skills = ["vibe"]
client_tools = ["read", "edit", "shell"]
# Coding needs far more tool calls than a chat turn; the default is 8.
limits = { max_iterations = 150, text_generation_timeout_seconds = 300 }
instructions = """
You build small, polished React + TypeScript + Vite + SCSS websites. Work only
inside /workspace. Inspect before editing, run the build, and finish with a
short factual summary. Chudbot commits and deploys after you stop.
"""

[bot.agents.vibe_coder.model]
id = "grok-4.3"
```

`tool_policy` defaults to `conversation` for existing subagents. A
`vibe_coder` agent runs under a `VibeCodingExecutor` that offers exactly the
three sandbox tools and none of the conversation, media, memory, or Discord
tools. The same agent cannot also be used as a normal conversation agent.
`check-config` warns when a `vibe_coder` agent keeps the default iteration
limit, since a coding job needs dozens of tool calls.
When the parent agent has an admitted Vibe binding it also gets
`vibe_check_names`, `vibe_list_sites`, and `vibe_manage`.

### Conversation tools

| Tool | What it does |
| --- | --- |
| `vibe` (subagent) | `{ action: "create" or "edit", siteName, task }`. Starts a job and returns the result and links. |
| `vibe_check_names` | Checks up to 8 candidate names. Returns `available`, `unavailable`, or `invalid` for each and nothing else. Reserves nothing. |
| `vibe_list_sites` | Lists active sites in this guild with name, description, access level, the actor's role, links, and last-changed time. Optional text filter. Current-conversation sites first, then the actor's own by recency, then the rest. |
| `vibe_manage` | `rollback` (to a revision number from `src.`, default the previous one), `add_editor`, `remove_editor`, `set_access`, `archive`, `restore`. Same server-side checks as everything else. `set_access` accepts `protected` or `public`. |

None of these accept a guild, user, or role argument. The actor comes from the
turn.

### Coding tools

| Tool | What it does |
| --- | --- |
| `read` | Reads a workspace-relative file (optional line range, numbered output) or lists a directory. Skips `.git`, `node_modules`, and `dist`. Returns images as image content. Cannot leave `/workspace`. |
| `edit` | Creates a file, deletes a file, or replaces one exact, unique string in a file. A missing or ambiguous match fails with a short message telling the model to reread. Cannot touch `.git`, `node_modules`, or `dist`. |
| `shell` | Runs one non-interactive command with `/bin/bash -lc` in the job's container, with a timeout and bounded output. Returns stdout, stderr, and the exit or timeout status. No PTY or background processes. |

`shell` covers `ls`, `rg`, `git diff`, `bun run build`, and tests. `read` and
`edit` exist because bounded reads and exact-match edits are cheaper and more
reliable than `cat` and `sed`, not for security. The agent can change anything
in the workspace with `shell`, and the export step treats the whole tree as
untrusted either way.

The model's prompt lists exactly these three tools. There is no LSP, browser,
screenshot, web fetch, or subagent tool in version 1. A screenshot-preview
tool is the most likely later addition.

### The job

```text
queued -> coding -> building -> committing -> done
            ^          |
            +- repair -+
```

1. Insert the job and site rows, check access, check out the current commit
   or template, and start the coding container.
2. Run the coding agent until it returns a final answer or hits a limit.
3. Export and validate the tree. An edit with no changes ends the job as
   `no_changes` with no commit.
4. Build in a fresh container. On a build or dependency failure, send the
   bounded log back to the same agent as another turn and go to step 2, at
   most `max_repair_attempts` times.
5. Commit, copy `dist/` into the artifact directory, insert the revision, set
   it active, and clear `running_job_id` in one transaction.
6. Remove the containers and workspace. Return links and the agent's summary
   to the conversation agent.

Terminal states: `done`, `no_changes`, `failed`, `cancelled`, `timed_out`.
Failures caused by access, cancellation, or sandbox problems never go back to
the model.

The job runs inline with the Discord turn. If that turns out to be too slow,
the same job rows can move behind a queue later without changing anything
else.

## `vibe.js` version 1

Served at `/__vibe/sdk/v1/vibe.js` on every site host and injected at build
time. Chudbot owns one canonical `vibe.d.ts`, serves it at
`/__vibe/sdk/v1/vibe.d.ts`, and writes `src/vibe.d.ts` when a coding workspace
opens and into the temporary clean-build tree. The generated file is excluded
from source export and removed after the build, so it is never committed or
shown in site source history. Existing sites therefore build against the
declarations compiled into the running Chudbot binary instead of retaining a
stale template copy.

```ts
type VibeIdentity = {
  id: string;            // Discord user id
  username: string;
  displayName: string;
  avatarUrl: string | null;
  guild: { id: string; displayName: string };
  site: { name: string };
};

const user = await vibe.identity(); // VibeIdentity | null
```

`identity()` does a same-origin `GET /__vibe/api/v1/identity`. On a protected
site, the server takes the site from the host and the user from the session
cookie, and returns nothing secret: no tokens, email, roles, or session id. On
a public site it returns JSON `null` and does not authenticate the request.
Errors use one envelope:

```json
{ "error": { "code": "not_authenticated", "message": "Sign in with Discord to continue." } }
```

`vibe.version` is `"1"`. Breaking changes go to `/v2/`; old sites keep `/v1/`.

### Site-local rooms

`vibe.room(name)` provides ephemeral WebSocket rooms. Every room is keyed by
the immutable Vibe site id and then by its room name; the client never supplies
a site id, and there is no cross-site discovery or messaging surface.

```ts
const room = vibe.room("topic");
room.on("message", (event, user) => console.log(event, user));
room.on("user:join", (user) => console.log("joined", user));
room.on("user:quit", (user) => console.log("left", user));
room.onUserState("presence", (user, before, after) => {
  console.log(user, before, after);
});

const connection = await room.join();
await connection.setUserState("presence", { status: "online" });
await connection.broadcast({ text: "hello" }, "message");
console.log(connection.self, connection.users);
await connection.disconnect();
```

The second argument to `broadcast` is the event label. `user:join`, `user:quit`,
and keyed user-state callbacks are implicit protocol events; application event
labels may not use the `user:` namespace. A logical user is present until their
last tab disconnects, so a second tab does not generate a duplicate join/quit
pair. User state lasts only for that presence and all room state disappears
when Chudbot restarts.

Event payloads and state values are arbitrary JSON: primitives such as `"abc"`,
arrays, and objects such as `{ "data": 123 }` all round-trip. The browser SDK
uses the platform JSON serializer and rejects values JSON cannot represent.

Protected rooms use the Discord membership identity already established for
the site request. Public rooms use a UUID generated by `crypto.randomUUID()`
and retained in origin-local `localStorage`; this is a display identity, not an
authentication credential. Both map to a small room user shape with an id,
display name, optional username/avatar, anonymous marker, and state map.

The room registry uses `DashMap` for sites and for the rooms within each site,
atomic deployment/site counters, and one short mutex per room to order that
room's membership and state transitions. Independent rooms on one site do not
share a mutex. Bounded message size, room/site/deployment connection counts,
room count, state keys/bytes, and per-connection outbound queues make excess
load fail locally. Per-connection command rate limiting bounds hot senders, and
a slow receiver is disconnected instead of accumulating an unbounded queue.

### Site-local database collections

Protected sites can persist JSON documents in collections scoped by the
immutable site id. Public sites cannot read, write, delete, count, or watch
collections; the backend returns a `collections_unavailable` API error even if
client code attempts it. The collection API uses the Discord-authenticated
session and additionally enforces a same-origin request.

```js
const scores = vibe.collection("scores");
const score = await scores.put({ user: "123", score: 10 });
const leaders = await scores
  .where({ user: ["123", "456"] })
  .orderBy("score", "desc")
  .limit(5)
  .find();
```

Documents receive `id`, `inserted_by`, `inserted_at`, `updated_by`, and
`updated_at`. `put` inserts without an id and updates an existing row when `id`
or `_id` is supplied. A supplied id that does not exist is an error. `insert`
never updates, while `update` requires an existing id. Supplied audit fields are
ignored and recomputed, so a returned document can be spread back into `put` or
`update`. JSON arrays and objects round-trip but are not queryable; `where`
compares only scalar columns against a scalar or list of scalars. Queries also
support offset, numeric/text/ISO timestamp ordering, projection, distinct
projected results, count, delete, and the multi-match-safe `deleteOne`.

The backend stores all collection rows in one Postgres JSONB table and performs
the deliberately unindexed document filtering in Rust. The site foreign key
uses `ON DELETE CASCADE`, so purging a site removes its collection data in the
same database transaction.

`watch(handler)` opens an authenticated WebSocket for the query's `where`
criteria. Insert events are emitted when the new document matches, delete
events when the old document matched, and update events when either side
matches. Events include the current document (or removed document for a
delete), `before`, `after`, `matchesBefore`, and `matchesAfter`, which lets a UI
recognize rows entering or leaving its live result set.

```js
const watch = await scores.where({ user: "123" }).watch((change) => {
  console.log(change.type, change.before, change.after);
});
await watch.disconnect();
```

Watch fanout is process-local Rust state with bounded queues and the existing
room connection limits. It is an ephemeral best-effort notification surface,
not a durable change log: reconnecting does not replay missed changes, and a
restart closes all watches.

## Serving sites

For protected sites, session and membership checks happen before routing. For
public sites, those checks are skipped. A request on `<name>.vibe.example` then
resolves as:

1. `/__vibe/*` is handled by Chudbot.
2. An exact file in the active revision's artifact directory is served.
3. Otherwise, if the request is a browser navigation (`Sec-Fetch-Dest:
   document`, or an `Accept` header with `text/html`), serve the root
   `index.html`. This makes React Router deep links and refreshes work.
4. Otherwise, 404.

Every response carries:

```text
Cache-Control: private, no-store
X-Content-Type-Options: nosniff
Referrer-Policy: no-referrer
X-Frame-Options: DENY
```

`no-store` matters for protected sites, where a Cloudflare cache hit would skip
the membership check. It is retained for public sites so an access change to
protected cannot leave public cached content behind. Content types come from
the file extension, not from anything in the site.

There is no Content-Security-Policy in version 1. It would stop sites from
calling public APIs, which is half the fun, and it does not protect against
the one threat we have accepted.

Login callbacks establish sessions from single-use bearer credentials and do
not mutate site data. Archived protected sites return a plain "not
available" page to members and a 404 to everyone else; an archived public site
returns the plain unavailable page.

Logging uses the existing tracing setup with the site name and job id on
spans. Cookies, OAuth codes, direct-login tokens, secrets, and source contents
are never logged.

## `src.vibe.example`

A read-only source browser built with the existing frontend tooling. It uses
the same session cookie and membership check as sites. For each site the
viewer may see, it shows the revision list with messages and authors, the
file tree and file contents at any revision, and diffs between revisions.
Source is rendered as escaped text, never executed. There are no buttons that
change anything; rollback, editors, and archiving happen in Discord through
`vibe_manage`.

## Cloudflare

The `vibe.example` zone has two proxied records pointing at one Cloudflare
Tunnel:

```text
@  CNAME  <tunnel-id>.cfargotunnel.com
*  CNAME  <tunnel-id>.cfargotunnel.com
```

The tunnel's ingress sends `vibe.example` and `*.vibe.example` to
`http://127.0.0.1:1860`. Cloudflare Universal SSL covers the apex and one
level of subdomains, which is exactly what Vibe uses. Creating a site never
touches Cloudflare.

Two zone settings are mandatory: HTTP redirects to HTTPS, and a Cache Rule
that bypasses cache for every request. Without the second one Cloudflare
could cache protected content or keep serving a formerly public site after it
is changed to protected.

`cloudflared` runs as its own systemd service, and its credential is not in
Chudbot config. Chudbot listens on loopback only, trusts forwarded headers
only on that path, and builds public URLs from `base_domain`, not from the
`Host` header. `check-config` refuses a Vibe-enabled config whose listener is
not loopback.

## Configuration

```toml
[vibe]
enabled = true
base_domain = "vibe.example"
root_dir = "vibe"
reserved_names = ["www", "api", "admin", "static", "status", "mail"]

[vibe.access]
# Restrict creating and editing to [bot].admins. Members can still view.
admins_only = false
# Empty means every guild the bot is in.
allowed_guilds = [
  { platform = "discord", guild_id = "123456789012345678" },
]

[vibe.auth]
client_id = "DISCORD_APPLICATION_ID"
client_secret = "DISCORD_OAUTH_CLIENT_SECRET"
session_days = 7

[vibe.sandbox]
docker_socket = "/var/run/docker.sock"
image = "chudbot-vibe-sandbox:latest"
job_timeout_seconds = 900
command_timeout_seconds = 60
build_timeout_seconds = 300
max_repair_attempts = 2
memory_mebibytes = 1024
cpus = 2
pids = 256

[vibe.limits]
max_source_files = 2000
max_file_bytes = 10485760
max_source_bytes = 52428800
max_artifact_bytes = 52428800
max_running_jobs_per_guild = 2
```

The OAuth callback is always `https://<base_domain>/oauth/callback`. Secrets
follow the existing config rules: redacted from diagnostics and never placed
in prompts, source, traces, or the SDK.

## Failure behavior

- **Discord unreachable**: cached membership results keep serving for up to 5
  minutes. Otherwise reads return 503 and jobs do not start.
- **Postgres unreachable**: nothing is served and no job starts.
- **Build fails**: the agent gets a repair turn. After the budget, the job
  fails and the previous revision stays live.
- **Job times out or is cancelled**: containers and workspace are removed. The
  site is unchanged, and a new site is deleted.
- **Crash mid-job**: on startup, Chudbot marks running jobs failed, removes
  their containers and workspaces, clears site locks, deletes `creating`
  sites, and repairs `main` in any repository where it lags Postgres.
- **Docker or image missing**: jobs fail with a clear error and sites keep
  serving.
- **Artifact directory missing for the active revision**: 503 and an error
  log. An owner or admin can roll back.
- **Cloudflare or tunnel down**: sites are unreachable until it recovers.
  Chudbot never opens a public listener as a fallback.

## Later, if wanted

Each of these is a separate small design when someone asks for it:

- a screenshot tool so the coding agent can see its work;
- per-site JSON storage (`vibe.db`) backed by Postgres JSONB with per-site
  quotas;
- `vibe.ai` calls with a per-site budget;
- posting to Discord from a site, with channel allowlists;
- ownership transfer, or a browser editor.

## Phases

Each phase updates `config.example.toml`, `check-config`, and `serve.sh`
alongside the behavior it adds.

1. **Sandbox, Git, and build**: `chudbot-api` contracts, tables, name rules,
   bare repositories, artifact storage, the `vibe-sandbox/` image and
   template, the two-container build, export validation, limits, the startup
   recovery pass, and skills config.
2. **Serving, login, and source browser**: host routing, Discord OAuth,
   sessions, the membership cache, site serving with SPA fallback and headers,
   the `vibe.example` explainer, `src.vibe.example`, and the Cloudflare and firewall
   runbook.
3. **Agents and jobs**: the `vibe_coder` policy and executor, the three coding
   tools, the `vibe` subagent, `vibe_check_names`, `vibe_list_sites`, the job
   state machine with repair turns, status messages, and the Discord reply.
   This is the first usable release.
4. **Management**: `vibe_manage` (rollback, editors, access level, archive,
   restore) and the `vibe purge` operator command.

## Tests

Tests mock Discord and model providers. Cloudflare is covered by a manual
smoke checklist.

- Names: the regex, reserved labels, `xn--`, batch checks, and two concurrent
  creates yielding one owner.
- Access: table-driven checks for member, editor, owner, admin, non-member,
  wrong guild, DM, `admins_only`, the guild allowlist, protected/public site
  access, and anonymous public identity.
- Login: state expiry and replay, `return` URL validation, cookie flags,
  logout, membership cache expiry, and Discord-down behavior.
- Host routing: apex, reserved labels, valid site, unknown site, nested
  labels, ports, trailing dots, and bad `Host` values.
- Serving: `/__vibe/` precedence, exact files, document fallback including
  paths with dots, 404 for other requests, and exact headers.
- Collections: protected/public gating, same-origin checks, automatic audit
  fields, write preconditions, scalar/list matching, JSON-column non-matches,
  ordering, pagination, projection/distinct/count, safe delete, purge cascade,
  and watch enter/leave/delete delivery with site/collection isolation.
- Export: property tests for path rules, symlinks, nested `.git`, reserved
  paths, size limits, and forbidden dependency sources.
- Sandbox: the container has no socket, secrets, or other workspaces; limits
  and timeouts are enforced; containers are removed on every exit path; the
  model never sees container ids or host paths.
- Tools: `read` ranges and directory listing; `edit` create, delete,
  unique-match replace, and failure messages; `shell` timeout, truncation,
  and non-zero exit reporting.
- Job: a final answer always leads to a clean build; no-diff edits make no
  commit; build failures produce repair turns up to the limit; access and
  cancellation errors never reach the model; a retried tool call returns the
  same job.
- Recovery: crash injection before the Git commit, after it, before the
  Postgres transaction, and after it. The previous revision is live in every
  case, and startup repairs `main` and locks.
- Config: rejects a non-loopback listener with Vibe enabled, a missing sandbox
  image, unknown skill names, and a `vibe_coder` agent used as a conversation
  agent.

## Done when

- A member asks for the mortgage example and gets working site and source
  links.
- The site is a Git repository from the template, built by the pinned image.
- An unnamed request gets a sensible available name without a clarifying
  question.
- "Change that site" works in a later conversation through
  `vibe_list_sites`.
- On a protected site, a non-member sees nothing: no HTML, assets, source, or
  identity.
- A member logs in with Discord once and can then open any site in their
  guilds.
- A public site skips OAuth, returns `null` from `vibe.identity()`, and keeps its
  source/history protected.
- On a protected site, `vibe.identity()` returns the viewer's name and guild,
  and nothing secret.
- Protected sites can persist and watch site-local collection documents;
  public sites receive an API error and purged sites retain no collection data.
- The owner and editors can edit and roll back; other members are refused.
- Two users racing for one name produce one owner.
- A failed, cancelled, or hostile coding run cannot change the live site or
  any other site.
- Agent shell commands run only inside the throwaway container, with no
  secrets or host paths.
- The coding model gets exactly `read`, `edit`, and `shell`. Commit and deploy
  happen only after it stops and the clean build passes.
- Sites survive a Chudbot redeploy and restart.
- `src.vibe.example` shows history, files, and diffs without executing anything.
- Cloudflare serves `vibe.example` and `*.vibe.example` through the tunnel with
  valid TLS and no cache hits.

## Runbook for the operator

These steps are outside the code and are done by hand before enabling Vibe:

1. Register `https://vibe.example/oauth/callback` on the Discord application and
   put the client id and secret in config.
2. Create the Cloudflare Tunnel, the two DNS records, the HTTPS redirect, and
   the bypass-cache rule.
3. Install Docker on the host and add the firewall rule blocking the Docker
   bridge from private ranges.
4. Run `serve.sh deploy`, then open the apex, `src.`, and a test site.
   Confirm protected login works, a non-member account is refused, public
   access works anonymously while source remains protected, and
   `CF-Cache-Status` never reports `HIT`.

## References

- [Bun install and lockfile documentation](https://bun.sh/docs/pm/cli/install)
- [Bun lifecycle-script policy](https://bun.sh/docs/pm/lifecycle)
- [Discord OAuth2 documentation](https://docs.discord.com/developers/topics/oauth2)
- [Discord guild resource: Get Guild Member](https://docs.discord.com/developers/resources/guild)
- [Cloudflare Universal SSL](https://developers.cloudflare.com/ssl/edge-certificates/universal-ssl/)
- [Cloudflare Tunnel ingress configuration](https://developers.cloudflare.com/tunnel/advanced/local-management/configuration-file/)
- [Cloudflare Cache Rules](https://developers.cloudflare.com/cache/how-to/cache-rules/create-dashboard/)
