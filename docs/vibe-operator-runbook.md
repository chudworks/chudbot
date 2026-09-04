# Vibe operator runbook (DGX Spark)

This is the production procedure for the Linux DGX Spark. It intentionally is
not automated by Chudbot. Run it only after reviewing the generated config and
firewall commands. The deployment uses the system Docker daemon at
`/var/run/docker.sock`; do not mount that socket into a Vibe container.

## 1. Discord application

1. In the Discord Developer Portal for the existing Chudbot application, add
   the exact OAuth2 redirect URI `https://vibe.example/oauth/callback`.
2. Copy the application id into `[vibe.auth].client_id` and create/copy the
   OAuth client secret into `[vibe.auth].client_secret` in the DGX
   `$CHUDBOT_DIR/config.toml`. Keep both values as quoted strings. Do not put the
   secret in the repository, shell history, or a site workspace.
3. Keep the OAuth scope at `identify`. Vibe uses the bot token, not a viewer's
   OAuth token, for current guild-membership checks.

## 2. System Docker and bridge firewall

1. Install Docker Engine from Docker's Ubuntu repository, enable the system
   service, and confirm that `docker info` uses the daemon socket
   `/var/run/docker.sock`. Add only the Unix account that runs Chudbot to the
   `docker` group, then start a fresh login session. Membership in that group is
   root-equivalent; do not grant it to site authors.
2. Apply the repository-managed rules to the dedicated sandbox Docker network:

   ```sh
   cd "$CHUDBOT_DIR/grok-discord-bot"
   ./serve.sh firewall-install
   ```

   The command idempotently creates the `chudbot-sandbox` Docker network with
   the fixed `chudbot-sbx0` bridge and owns one `CHUDBOT_SANDBOX` chain. Other
   Docker networks are unaffected. It rejects loopback, link-local,
   carrier-grade NAT, RFC 1918, and IPv6 private destinations while allowing
   public Bun/npm traffic.
3. Verify the Docker network and every managed firewall rule:

   ```sh
   ./serve.sh firewall-check
   ```

   `serve.sh deploy` calls `firewall-install` on every deploy, which also
   restores rules after a reboot. No systemd unit or global helper is
   installed. `./serve.sh firewall-remove` stops Chudbot, refuses to proceed if
   a Vibe container remains attached, and then removes the rules and network.
4. Build the image, then inspect a throwaway container. It must reach
   `https://registry.npmjs.org/`, must not reach a known LAN address or the
   Docker bridge gateway, and must contain no `/var/run/docker.sock`, Chudbot
   config, media, repositories, artifacts, or sibling workspaces. Verify the
   configured non-root user, read-only root filesystem, dropped capabilities,
   no-new-privileges, 1 GiB memory, 2 CPU, and 256 PID limits with
   `docker inspect`.

## 3. Cloudflare zone and tunnel

1. Add `vibe.example` to Cloudflare and create one named Tunnel. Install
   `cloudflared` on the DGX as its own systemd service; keep its credential in
   the cloudflared service configuration, never Chudbot config.
2. Configure exactly two proxied DNS records pointing to
   `<tunnel-id>.cfargotunnel.com`: apex `@` and wildcard `*`, both CNAME.
3. Configure tunnel ingress for both `vibe.example` and `*.vibe.example` to
   `http://127.0.0.1:1860`, followed by the normal terminal 404 rule. Confirm
   Chudbot's `[web].listen` contains loopback addresses only.
4. Enable “Always Use HTTPS” (or an equivalent redirect rule) for the zone.
5. Create a Cache Rule matching every hostname in the zone and set cache
   eligibility to bypass. Do not rely only on origin `Cache-Control`; the rule
   prevents a Cloudflare cache hit from bypassing a membership check or
   retaining public content after a site becomes protected.
6. Confirm Universal SSL is active for the apex and one-level wildcard before
   testing a site.

## 4. Configure, validate, migrate, and deploy

1. Back up Postgres and `$CHUDBOT_DIR/vibe` together. Restore them as one unit.
2. Copy the `[vibe]` example into the production config. Set the intended
   Discord guild string ids, rollout switches, OAuth credentials, and keep
   `docker_socket = "/var/run/docker.sock"`.
   Also copy both `[bot.skills.vibe]` and `[bot.skills.vibe_conversation]`
   tables plus the `vibe_coder` agent/binding from `config.example.toml`.
   `serve.sh deploy` installs the referenced Markdown files atomically into
   `$CHUDBOT_DIR/skills` before `check-config` runs.
3. Build the exact local sandbox image, build the new binary, and run its
   config check before stopping the installed service:

   ```sh
   cd "$CHUDBOT_DIR/grok-discord-bot"
   docker build --pull --tag chudbot-vibe-sandbox:latest vibe-sandbox
   cargo build --locked --profile distribute -p chudbot-bin
   target/distribute/chudbot --config "$CHUDBOT_DIR/config.toml" check-config
   ```

   The check must confirm a loopback listener, valid string ids, readable
   skills, and the locally built sandbox image.
4. Review the pending migration, stop Chudbot, run `chudbot migrate` once, and
   then run `serve.sh deploy`. The script builds the pinned sandbox locally,
   atomically refreshes the template, and preserves `$CHUDBOT_DIR/vibe`.

## 5. Required smoke test

1. Open `https://vibe.example/` and `https://src.vibe.example/`; confirm valid TLS and
   that a site/source request redirects to Discord login only for a document
   navigation. Asset and API requests without a session must return 401.
2. Create a disposable site from an allowed guild. Confirm the Discord reply's
   Site and Source links, the Git history, the clean-build revision, an SPA deep
   link refresh, `vibe.identity()`'s exact public fields, and that
   `/__vibe/sdk/v1/vibe.d.ts` is available but `src/vibe.d.ts` is absent from
   the committed/source-browser tree.
3. With a second account that is not in the guild, request the HTML, a known
   asset, source, and the identity API. Every request must reveal no site data.
4. Make the site public. Confirm a signed-out browser can load the HTML and
   assets without an OAuth redirect, `vibe.identity()` returns JSON `null`, and
   the source link still requires guild membership. Set it back to
   `🔒 protected` and confirm the OAuth redirect returns.
5. Ask Chudbot for a login link for the requesting member and for a mentioned
   second member. Confirm only each target's DM receives its link, each link
   expires after 10 minutes or one use, redemption skips Discord OAuth, and the
   success page greets the target member rather than redirecting to `/`. Confirm
   the resulting cookie uses `[vibe.auth].session_days`, a nonmember target is
   rejected, and failed DM delivery leaves no redeemable link.
6. Add an editor, edit, roll back, archive, restore, and have another user race
   for the same new name. Confirm the permissions and single winner.
7. Force a build failure and cancel a coding run. Confirm the previous revision
   remains live, no partial artifact is served, and containers/workspaces are
   removed. Restart Chudbot during a disposable job and confirm recovery clears
   locks and repairs `main` to the Postgres-active commit.
8. Inspect response headers on HTML, JS, images, SDK, and identity responses:
   `Cache-Control: private, no-store`, `X-Content-Type-Options: nosniff`,
   `Referrer-Policy: no-referrer`, and `X-Frame-Options: DENY`.
9. Inspect Cloudflare responses repeatedly. `CF-Cache-Status` must never be
   `HIT` for the apex, source, any site file, or `/__vibe/` response.
10. Open two tabs on one protected site and exercise room join, broadcast,
   user-state, and quit events. Repeat on a public site and confirm its anonymous
   id survives a reload. Join the same room name on a second site and confirm no
   users, state, or events cross the site boundary.
11. On a protected site, insert, update, filter, count, and delete collection
    documents, including a JSON-valued non-matching column and `deleteOne` with
    multiple matches. Open a watch in a second tab and confirm insert, update
    entry/exit, and delete events. Make the site public and confirm collection
    HTTP and WebSocket endpoints return `collections_unavailable`. Purge the
    disposable site and confirm its `vibe_collection_documents` rows cascade.
