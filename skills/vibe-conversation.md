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
ask the model to provide an owning guild, owner, user id, role, commit, or host
path. A target guild id is accepted only for an explicit site-sharing request.
Only claim that a site was created, edited, or deployed when the corresponding
tool result is successful. A failed result means the live site was not changed.
For `access_denied`, tell the requester that the site was not modified and that
the owner or a Chudbot admin must authorize them as an editor; never describe
the requested changes as queued, locked in, or completed.

The site owner or a Chudbot admin can authorize another current server member
with `vibe_manage` and `action: "add_editor"`, and revoke that access with
`action: "remove_editor"`. Use the numeric user id from trusted message context,
such as a mentioned user's id; never derive an id from a display name or
username. Editor access permits edits and rollbacks, but not editor management,
access-level changes, archiving, or restoration.

New sites are `🔒 protected`: deployed-site viewers must sign in with Discord
and remain members of the owning server or a server explicitly added to the
site. When the owner or a Chudbot admin asks to share a protected site with
another server, use `vibe_manage` with `action: "add_guild"` and the supplied
numeric `guildId`; never infer an id from a server name. This grants current
members view access to the deployed site and source, but no edit access. Use
`vibe_manage` with
`action: "set_access"` and `accessLevel: "public"` or `"protected"` only when
the user asks to change who can view the deployed site. Public access does not
publish the source/history browser, but it disables the site-local collection
API and collection watches. If a request requires durable collection data and
public access at the same time, explain that conflict instead of promising both.
State the resulting access label clearly.

When a current server member asks for a Vibe login, sign-in, or auth link, use
`send_vibe_login_link`; omit `userId` for the requester or copy a mentioned
member's numeric id from trusted message context to deliver it to someone else.
The secret, single-use link is sent only to the target member's DM. Never put a
login link or session token in the public reply or ask the recipient to paste it
back into Discord.
