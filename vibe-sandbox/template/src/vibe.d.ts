type VibeIdentity = {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
  guild: { id: string; displayName: string };
  site: { name: string };
};

declare const vibe: {
  readonly version: "1";
  identity(): Promise<VibeIdentity | null>;
};
