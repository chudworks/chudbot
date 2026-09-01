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
2. Confirm forwarding is managed through the `DOCKER-USER` chain:

   ```sh
   sudo iptables -S DOCKER-USER
   sudo ip6tables -S DOCKER-USER
   ```

3. Insert the following rules once. They preserve established connections and
   reject new traffic arriving from Docker bridge interfaces toward loopback,
   link-local, carrier-grade NAT, and RFC 1918 networks. Public npm/Bun traffic
   remains allowed.

   ```sh
   sudo iptables -I DOCKER-USER 1 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
   sudo iptables -I DOCKER-USER 2 -i docker0 -d 127.0.0.0/8 -j REJECT
   sudo iptables -I DOCKER-USER 3 -i docker0 -d 169.254.0.0/16 -j REJECT
   sudo iptables -I DOCKER-USER 4 -i docker0 -d 100.64.0.0/10 -j REJECT
   sudo iptables -I DOCKER-USER 5 -i docker0 -d 10.0.0.0/8 -j REJECT
   sudo iptables -I DOCKER-USER 6 -i docker0 -d 172.16.0.0/12 -j REJECT
   sudo iptables -I DOCKER-USER 7 -i docker0 -d 192.168.0.0/16 -j REJECT
   sudo iptables -I DOCKER-USER 8 -i br+ -d 127.0.0.0/8 -j REJECT
   sudo iptables -I DOCKER-USER 9 -i br+ -d 169.254.0.0/16 -j REJECT
   sudo iptables -I DOCKER-USER 10 -i br+ -d 100.64.0.0/10 -j REJECT
   sudo iptables -I DOCKER-USER 11 -i br+ -d 10.0.0.0/8 -j REJECT
   sudo iptables -I DOCKER-USER 12 -i br+ -d 172.16.0.0/12 -j REJECT
   sudo iptables -I DOCKER-USER 13 -i br+ -d 192.168.0.0/16 -j REJECT
   sudo ip6tables -I DOCKER-USER 1 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
   sudo ip6tables -I DOCKER-USER 2 -i docker0 -d ::1/128 -j REJECT
   sudo ip6tables -I DOCKER-USER 3 -i docker0 -d fe80::/10 -j REJECT
   sudo ip6tables -I DOCKER-USER 4 -i docker0 -d fc00::/7 -j REJECT
   sudo ip6tables -I DOCKER-USER 5 -i br+ -d ::1/128 -j REJECT
   sudo ip6tables -I DOCKER-USER 6 -i br+ -d fe80::/10 -j REJECT
   sudo ip6tables -I DOCKER-USER 7 -i br+ -d fc00::/7 -j REJECT
   ```

4. Persist the rules with the host's normal firewall mechanism (on Ubuntu,
   `iptables-persistent`/`netfilter-persistent` is suitable), reboot once, and
   verify the rules still appear before enabling Vibe.
5. Build the image, then inspect a throwaway container. It must reach
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
   prevents a Cloudflare cache hit from bypassing a membership check.
6. Confirm Universal SSL is active for the apex and one-level wildcard before
   testing a site.

## 4. Configure, validate, migrate, and deploy

1. Back up Postgres and `$CHUDBOT_DIR/vibe` together. Restore them as one unit.
2. Copy the `[vibe]` example into the production config. Set the intended
   Discord guild string ids, rollout switches, OAuth credentials, and keep
   `docker_socket = "/var/run/docker.sock"`.
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
   link refresh, and `vibe.identity()`'s exact public fields.
3. With a second account that is not in the guild, request the HTML, a known
   asset, source, and the identity API. Every request must reveal no site data.
4. Add an editor, edit, roll back, archive, restore, and have another user race
   for the same new name. Confirm the permissions and single winner.
5. Force a build failure and cancel a coding run. Confirm the previous revision
   remains live, no partial artifact is served, and containers/workspaces are
   removed. Restart Chudbot during a disposable job and confirm recovery clears
   locks and repairs `main` to the Postgres-active commit.
6. Inspect response headers on HTML, JS, images, SDK, and identity responses:
   `Cache-Control: private, no-store`, `X-Content-Type-Options: nosniff`,
   `Referrer-Policy: no-referrer`, and `X-Frame-Options: DENY`.
7. Inspect Cloudflare responses repeatedly. `CF-Cache-Status` must never be
   `HIT` for the apex, source, any site file, or `/__vibe/` response.
