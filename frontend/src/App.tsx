import { useAtom, useAtomValue, useSetAtom } from "jotai";
import {
  Languages,
  Loader2,
  Mic,
  MicOff,
  Plug,
  Radio,
  Square,
  Trash2,
  Waves,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import {
  configAtom,
  connectionStatusAtom,
  errorAtom,
  interimAtom,
  isRecordingAtom,
  messagesAtom,
  statsAtom,
  type AppConfig,
  type TranscriptMessage,
} from "@/state/disco";

type ServerMessage =
  | ({ type: "config" } & AppConfig)
  | {
      type: "interim";
      text: string;
      span?: [number, number];
      utterance_id?: number;
      speaker?: number;
      translation?: string;
    }
  | {
      type: "final";
      text: string;
      span?: [number, number];
      utterance_id?: number;
      translation?: string;
      speaker?: number;
    }
  | { type: "pong" };

const speakerPalette = [
  "bg-sky-500/15 text-sky-200 ring-sky-400/25",
  "bg-rose-500/15 text-rose-200 ring-rose-400/25",
  "bg-emerald-500/15 text-emerald-200 ring-emerald-400/25",
  "bg-amber-500/15 text-amber-100 ring-amber-300/25",
];

function speakerLabel(speaker?: number) {
  return speaker === undefined ? "S?" : `S${speaker}`;
}

function speakerClass(speaker?: number) {
  if (speaker === undefined) {
    return "bg-zinc-500/15 text-zinc-300 ring-zinc-400/25";
  }
  return speakerPalette[speaker % speakerPalette.length];
}

function wsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

function useDiscoSocket() {
  const setConfig = useSetAtom(configAtom);
  const setConnection = useSetAtom(connectionStatusAtom);
  const setMessages = useSetAtom(messagesAtom);
  const setInterim = useSetAtom(interimAtom);
  const setError = useSetAtom(errorAtom);
  const retryRef = useRef<number | null>(null);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;

    function connect() {
      setConnection("connecting");
      socket = new WebSocket(wsUrl());

      socket.onopen = () => {
        setConnection("connected");
        setError(null);
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data) as ServerMessage;
        if (data.type === "config") {
          setConfig({
            language: data.language,
            translate_korean: data.translate_korean,
            is_recording: data.is_recording,
          });
          return;
        }
        if (data.type === "interim") {
          setInterim((current) => {
            if (
              data.translation &&
              current?.utterance_id !== undefined &&
              data.utterance_id !== undefined &&
              current.utterance_id !== data.utterance_id
            ) {
              return current;
            }

            const incomingStart = data.span?.[0];
            const incomingEnd = data.span?.[1] ?? Number.POSITIVE_INFINITY;
            const currentStart = current?.span?.[0];
            const currentEnd = current?.span?.[1] ?? Number.NEGATIVE_INFINITY;
            const isNewUtterance =
              current !== null &&
              ((data.utterance_id !== undefined &&
                current.utterance_id !== undefined &&
                data.utterance_id !== current.utterance_id) ||
                (incomingStart !== undefined &&
                currentStart !== undefined &&
                Math.abs(incomingStart - currentStart) > 0.25) ||
                (data.speaker !== undefined &&
                  current.speaker !== undefined &&
                  data.speaker !== current.speaker) ||
                (!data.translation &&
                  current.text.length > 0 &&
                  data.text.length + 3 < current.text.length));

            if (
              data.translation &&
              current &&
              !isNewUtterance &&
              incomingEnd < currentEnd
            ) {
              return {
                ...current,
                translation: data.translation,
              };
            }

            return {
              text: data.text,
              span: data.span,
              utterance_id: data.utterance_id,
              speaker: data.speaker,
              translation: data.translation ?? (isNewUtterance ? undefined : current?.translation),
            };
          });
          return;
        }
        if (data.type === "final") {
          setInterim((current) => {
            if (
              current?.utterance_id !== undefined &&
              data.utterance_id !== undefined &&
              current.utterance_id !== data.utterance_id
            ) {
              return current;
            }
            return null;
          });
          setMessages((current) => [
            ...current,
            {
              id:
                data.utterance_id !== undefined
                  ? String(data.utterance_id)
                  : crypto.randomUUID(),
              text: data.text,
              translation: data.translation,
              span: data.span,
              utterance_id: data.utterance_id,
              speaker: data.speaker,
              time: new Date().toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              }),
            },
          ]);
        }
      };

      socket.onerror = () => {
        setError("WebSocket connection failed.");
      };

      socket.onclose = () => {
        setConnection("disconnected");
        if (!closed) {
          retryRef.current = window.setTimeout(connect, 2000);
        }
      };
    }

    connect();

    return () => {
      closed = true;
      if (retryRef.current !== null) {
        window.clearTimeout(retryRef.current);
      }
      socket?.close();
    };
  }, [setConfig, setConnection, setError, setInterim, setMessages]);
}

function StatusBadge() {
  const connection = useAtomValue(connectionStatusAtom);
  const recording = useAtomValue(isRecordingAtom);

  if (connection !== "connected") {
    return (
      <Badge variant="secondary" className="gap-1.5 text-zinc-300">
        <Plug className="size-3.5" />
        {connection}
      </Badge>
    );
  }

  return (
    <Badge className={cn("gap-1.5", recording && "bg-rose-500 text-white")}>
      {recording ? <Radio className="size-3.5" /> : <Plug className="size-3.5" />}
      {recording ? "recording" : "connected"}
    </Badge>
  );
}

function MessageRow({
  message,
  compact,
}: {
  message: TranscriptMessage;
  compact: boolean;
}) {
  return (
    <article
      className={cn(
        "border-l-2 border-zinc-700 bg-zinc-950/55 px-4 py-3 shadow-sm ring-1 ring-white/5",
        compact ? "rounded-md" : "rounded-lg",
        message.speaker === 0 && "border-sky-400",
        message.speaker === 1 && "border-rose-400",
        message.speaker === 2 && "border-emerald-400",
        message.speaker === 3 && "border-amber-300",
      )}
    >
      {!compact && (
        <div className="mb-2 flex items-center gap-2">
          <span
            className={cn(
              "inline-flex h-6 items-center rounded-md px-2 text-xs font-medium ring-1",
              speakerClass(message.speaker),
            )}
          >
            {speakerLabel(message.speaker)}
          </span>
          <span className="text-xs text-zinc-500">{message.time}</span>
        </div>
      )}
      <p className="text-[0.95rem] leading-6 text-zinc-100">{message.text}</p>
      {message.translation && (
        <>
          <Separator className="my-3 bg-zinc-800" />
          <p className="text-[0.92rem] leading-6 text-cyan-200">
            {message.translation}
          </p>
        </>
      )}
    </article>
  );
}

function TranscriptPanel() {
  const messages = useAtomValue(messagesAtom);
  const interim = useAtomValue(interimAtom);
  const viewportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const viewport = viewportRef.current?.querySelector(
      "[data-radix-scroll-area-viewport]",
    );
    viewport?.scrollTo({ top: viewport.scrollHeight, behavior: "smooth" });
  }, [messages, interim]);

  return (
    <Card className="min-h-0 flex-1 overflow-hidden border-zinc-800 bg-zinc-950/70">
      <CardHeader className="flex-row items-center justify-between gap-4 border-b border-zinc-800 px-4 py-3">
        <CardTitle className="text-sm font-medium text-zinc-200">
          Transcript
        </CardTitle>
        <Badge variant="outline" className="gap-1.5 border-zinc-700 text-zinc-400">
          <Waves className="size-3.5" />
          live
        </Badge>
      </CardHeader>
      <CardContent className="h-full min-h-0 p-0">
        <ScrollArea ref={viewportRef} className="h-[calc(100vh-13.5rem)]">
          <div className="flex flex-col gap-3 p-4">
            {messages.length === 0 && !interim && (
              <div className="grid min-h-[22rem] place-items-center text-center">
                <div>
                  <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900">
                    <Mic className="size-5 text-zinc-400" />
                  </div>
                  <p className="text-sm font-medium text-zinc-300">
                    Waiting for speech
                  </p>
                  <p className="mt-1 text-xs text-zinc-500">
                    Start recording to stream transcript turns.
                  </p>
                </div>
              </div>
            )}
            {messages.map((message, index) => {
              const previous = messages[index - 1];
              const compact =
                previous?.speaker !== undefined &&
                message.speaker !== undefined &&
                previous.speaker === message.speaker;
              return (
                <MessageRow key={message.id} message={message} compact={compact} />
              );
            })}
            {interim && (
              <article className="rounded-lg border border-dashed border-cyan-500/30 bg-cyan-500/5 px-4 py-3">
                <div className="mb-2 flex items-center gap-2">
                  <span
                    className={cn(
                      "inline-flex h-6 items-center rounded-md px-2 text-xs font-medium ring-1",
                      speakerClass(interim.speaker),
                    )}
                  >
                    {speakerLabel(interim.speaker)}
                  </span>
                  <Loader2 className="size-3.5 animate-spin text-cyan-300" />
                </div>
                <p className="text-[0.95rem] leading-6 text-zinc-300">
                  {interim.text}
                </p>
                {interim.translation && (
                  <>
                    <Separator className="my-3 bg-cyan-500/20" />
                    <p className="text-[0.92rem] leading-6 text-cyan-200">
                      {interim.translation}
                    </p>
                  </>
                )}
              </article>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function SidePanel() {
  const config = useAtomValue(configAtom);
  const stats = useAtomValue(statsAtom);

  const rows = useMemo(
    () => [
      { label: "Turns", value: stats.turns },
      { label: "Speakers", value: stats.speakers },
      { label: "Translated", value: stats.translated },
    ],
    [stats],
  );

  return (
    <aside className="grid gap-3 lg:w-72">
      <Card className="border-zinc-800 bg-zinc-950/70">
        <CardHeader className="px-4 py-3">
          <CardTitle className="text-sm font-medium text-zinc-200">
            Session
          </CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3 px-4 pb-4">
          <div className="flex items-center justify-between text-sm">
            <span className="text-zinc-500">Language</span>
            <Badge variant="secondary">{config.language}</Badge>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-zinc-500">Korean</span>
            <Badge variant={config.translate_korean ? "default" : "outline"}>
              {config.translate_korean ? "on" : "off"}
            </Badge>
          </div>
        </CardContent>
      </Card>

      <Card className="border-zinc-800 bg-zinc-950/70">
        <CardHeader className="px-4 py-3">
          <CardTitle className="text-sm font-medium text-zinc-200">
            Activity
          </CardTitle>
        </CardHeader>
        <CardContent className="grid gap-2 px-4 pb-4">
          {rows.map((row) => (
            <div
              key={row.label}
              className="flex items-center justify-between rounded-md bg-zinc-900/70 px-3 py-2"
            >
              <span className="text-xs text-zinc-500">{row.label}</span>
              <span className="text-sm font-medium text-zinc-100">{row.value}</span>
            </div>
          ))}
        </CardContent>
      </Card>
    </aside>
  );
}

function ControlBar() {
  const [recording, setRecording] = useAtom(isRecordingAtom);
  const setMessages = useSetAtom(messagesAtom);
  const setInterim = useSetAtom(interimAtom);
  const [error, setError] = useAtom(errorAtom);
  const [pending, setPending] = useState(false);

  async function toggleRecording(next: boolean) {
    setPending(true);
    setError(null);
    try {
      const response = await fetch(next ? "/api/start" : "/api/stop", {
        method: "POST",
      });
      if (!response.ok) {
        const data = (await response.json().catch(() => null)) as
          | { detail?: string }
          | null;
        throw new Error(data?.detail ?? "Request failed");
      }
      const data = (await response.json()) as { status: string };
      if (data.status === "started") {
        setRecording(true);
      }
      if (data.status === "stopped") {
        setRecording(false);
        setInterim(null);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Request failed");
    } finally {
      setPending(false);
    }
  }

  return (
    <footer className="border-t border-zinc-800 bg-zinc-950/90 px-4 py-3">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3 text-sm text-zinc-400">
          <div
            className={cn(
              "flex size-8 items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900",
              recording && "border-rose-500/40 bg-rose-500/10 text-rose-200",
            )}
          >
            {recording ? <Mic className="size-4" /> : <MicOff className="size-4" />}
          </div>
          <span>{recording ? "Listening" : "Recorder stopped"}</span>
          {error && <span className="text-rose-300">{error}</span>}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => {
              setMessages([]);
              setInterim(null);
            }}
          >
            <Trash2 className="size-4" />
            Clear
          </Button>
          <Button
            variant={recording ? "destructive" : "default"}
            disabled={pending}
            onClick={() => void toggleRecording(!recording)}
          >
            {pending ? (
              <Loader2 className="size-4 animate-spin" />
            ) : recording ? (
              <Square className="size-4" />
            ) : (
              <Radio className="size-4" />
            )}
            {recording ? "Stop" : "Start"}
          </Button>
        </div>
      </div>
    </footer>
  );
}

export default function App() {
  useDiscoSocket();
  const config = useAtomValue(configAtom);

  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="border-b border-zinc-800 bg-zinc-950/95 px-4 py-3">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-lg bg-cyan-400/10 text-cyan-200 ring-1 ring-cyan-300/20">
              <Waves className="size-5" />
            </div>
            <div>
              <h1 className="text-base font-semibold tracking-normal text-zinc-50">
                Disco
              </h1>
              <p className="text-xs text-zinc-500">Realtime ASR console</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="gap-1.5 border-zinc-700">
              <Languages className="size-3.5" />
              {config.language}
            </Badge>
            <StatusBadge />
          </div>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-7xl flex-1 gap-3 overflow-hidden p-4">
        <TranscriptPanel />
        <div className="hidden lg:block">
          <SidePanel />
        </div>
      </main>

      <ControlBar />
    </div>
  );
}
