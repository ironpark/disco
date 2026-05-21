import { atom } from "jotai";

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

export type AppConfig = {
  language: string;
  translate_korean: boolean;
  is_recording: boolean;
};

export type TranscriptMessage = {
  id: string;
  text: string;
  translation?: string;
  span?: [number, number];
  utterance_id?: number;
  utterance_ids?: number[];
  speaker?: number;
  time: string;
};

export type InterimMessage = {
  text: string;
  span?: [number, number];
  utterance_id?: number;
  speaker?: number;
  translation?: string;
};

export type InterimState = {
  byId: Record<string, InterimMessage>;
  fallback: InterimMessage[];
};

export const configAtom = atom<AppConfig>({
  language: "-",
  translate_korean: false,
  is_recording: false,
});

export const connectionStatusAtom = atom<ConnectionStatus>("connecting");
export const messagesAtom = atom<TranscriptMessage[]>([]);
export const interimAtom = atom<InterimState>({ byId: {}, fallback: [] });
export const errorAtom = atom<string | null>(null);

export const isRecordingAtom = atom(
  (get) => get(configAtom).is_recording,
  (get, set, next: boolean) => {
    set(configAtom, { ...get(configAtom), is_recording: next });
  },
);

export const statsAtom = atom((get) => {
  const messages = get(messagesAtom);
  const speakers = new Set(
    messages
      .map((message) => message.speaker)
      .filter((speaker): speaker is number => speaker !== undefined),
  );
  return {
    turns: messages.length,
    speakers: speakers.size,
    translated: messages.filter((message) => message.translation).length,
  };
});

export const interimListAtom = atom((get) => {
  const interim = get(interimAtom);
  return [...Object.values(interim.byId), ...interim.fallback].sort(
    (left, right) => (left.span?.[0] ?? 0) - (right.span?.[0] ?? 0),
  );
});
