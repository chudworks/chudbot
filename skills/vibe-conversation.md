# Vibe conversation workflow

Use `vibe` with `create` only for a new site and `edit` only for an existing
site. Never turn a failed create into an edit. If the user supplies a name,
keep it exactly and report `name_unavailable` with alternatives if claimed.
If no name is supplied, invent several short descriptive DNS-label candidates,
check up to eight with `vibe_check_names`, and choose the best available name
without asking.

For references such as “that site”, use `vibe_list_sites`. Prefer sites from
the current conversation, then the user's own recent sites, then other guild
sites. Ask when two results are genuinely plausible. Do not guess across
guilds.

After a successful job, report both the Site and Source links returned by the
tool. Chudbot owns the actor, Git commit, clean build, and deployment; never
ask the model to provide a guild, owner, user id, role, commit, or host path.

New sites are `🔒 protected`: deployed-site viewers must sign in with Discord
and remain members of the owning server. Use `vibe_manage` with
`action: "set_access"` and `accessLevel: "public"` or `"protected"` only when
the user asks to change who can view the deployed site. Public access does not
publish the source/history browser, but it disables the site-local collection
API and collection watches. If a request requires durable collection data and
public access at the same time, explain that conflict instead of promising both.
State the resulting access label clearly.
