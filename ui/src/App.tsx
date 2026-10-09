import { useEffect, useState } from "react";
import {
    ArrowDownRight,
    ArrowUpRight,
    ArrowUpRight as External,
    Check,
    RefreshCw,
    Search,
    Trophy,
} from "lucide-react";
import "./styles.css";
import ceruleanLogo from "./assets/cerulean.png";
import labNames from "./lab-names.json";

type Entry = {
    rank: number;
    github_actor: string;
    score: number | null;
    submitted_at: string;
    run_url: string | null;
    submission_id: string;
    attempts: number;
};
type Lab = {
    id: string;
    grading_type: "pass_fail" | "score";
    primary_metric: string | null;
    metric_direction: string | null;
    entries: Entry[];
    submissions: number;
    participants: number;
};
type Snapshot = {
    source: string;
    updated_at: string | null;
    error: string | null;
    labs: Lab[];
};
const names: Record<string, string> = labNames;
const date = (value: string) =>
    new Date(value).toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });
function runLink(url: string | null) {
    try {
        return url && new URL(url).protocol === "https:" ? url : null;
    } catch {
        return null;
    }
}

function GitHubAvatar({ username }: { username: string }) {
    const [failed, setFailed] = useState(false);
    return (
        <span className="avatar" aria-hidden="true">
            {failed ? (
                username.slice(0, 2).toUpperCase()
            ) : (
                <img
                    src={`https://github.com/${encodeURIComponent(username)}.png?size=72`}
                    alt=""
                    width={36}
                    height={36}
                    loading="lazy"
                    decoding="async"
                    referrerPolicy="no-referrer"
                    onError={() => setFailed(true)}
                />
            )}
        </span>
    );
}

export default function App() {
    const [data, setData] = useState<Snapshot | null>(null);
    const [selected, setSelected] = useState("");
    const [query, setQuery] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    useEffect(() => {
        const controller = new AbortController();
        const load = async () => {
            setLoading(true);
            try {
                const response = await fetch("/api/leaderboard", {
                    signal: controller.signal,
                });
                if (!response.ok)
                    throw new Error(
                        "Leaderboard unavailable. Please try again.",
                    );
                const snapshot: Snapshot = await response.json();
                setData(snapshot);
                setSelected(
                    (previous) => previous || [...snapshot.labs].sort(
                        (a, b) => b.id.localeCompare(a.id, undefined, { numeric: true }),
                    )[0]?.id || "",
                );
                setError("");
            } catch (e) {
                if (!controller.signal.aborted)
                    setError(
                        e instanceof Error
                            ? e.message
                            : "Unable to load standings.",
                    );
            } finally {
                if (!controller.signal.aborted) setLoading(false);
            }
        };
        void load();
        const interval = setInterval(() => void load(), 30_000);
        return () => {
            controller.abort();
            clearInterval(interval);
        };
    }, []);
    const lab = data?.labs.find((item) => item.id === selected);
    const scored = lab?.grading_type === "score";
    const entries =
        lab?.entries.filter((entry) =>
            entry.github_actor.toLowerCase().includes(query.toLowerCase()),
        ) ?? [];
    return (
        <div className="shell">
            <header className="nav">
                <a href="/" className="brand">
                    <span className="brand-mark">
                        <img src={ceruleanLogo} alt="" />
                    </span>{" "}
                    Cerulean Labs
                </a>
                <a
                    className="nav-link"
                    href="https://wandb.ai/cerulean-labs/study-group-labs"
                    target="_blank"
                    rel="noreferrer"
                >
                    View experiments <External size={16} />
                </a>
            </header>
            <main>
                <section className="hero">
                    <h1>
                        Small experiments. <em>Better, together.</em>
                    </h1>
                </section>
                <div className="workspace">
                    <div className="lab-toolbar" id="standings">
                        <aside aria-label="Choose a lab">
                            {data?.labs.map((item) => (
                                <button
                                    key={item.id}
                                    aria-pressed={selected === item.id}
                                    className={`lab-button ${selected === item.id ? "active" : ""}`}
                                    onClick={() => {
                                        setSelected(item.id);
                                        setQuery("");
                                    }}
                                >
                                    <span className="lab-number">
                                        {item.id
                                            .replace("lab", "")
                                            .padStart(2, "0")}
                                    </span>
                                    <strong>
                                        {item.id.replace("lab", "Lab ")}
                                    </strong>
                                    <small>
                                        {names[item.id] ??
                                            "Study group experiment"}
                                    </small>
                                </button>
                            ))}
                        </aside>
                        <span className="sync">
                            <RefreshCw
                                size={13}
                                className={loading ? "spinning" : ""}
                            />
                            {data?.updated_at
                                ? `Updated ${date(data.updated_at)}`
                                : "Waiting for first sync"}
                        </span>
                    </div>
                    <section className="board" aria-label="Leaderboard">
                        <div className="board-header">
                            <div>
                                <div className="eyebrow accent">
                                    {lab?.id.replace("lab", "LAB ") ??
                                        "STANDINGS"}
                                </div>
                                <h2>
                                    {lab
                                        ? (names[lab.id] ?? lab.id)
                                        : "Loading experiments"}
                                </h2>
                                <p>
                                    {scored
                                        ? `${lab?.metric_direction === "maximize" ? "Highest" : "Lowest"} ${lab?.primary_metric} per participant. Keep improving.`
                                        : "Correctness first. Ranked by earliest passing submission."}
                                </p>
                            </div>
                            <span className="mode">
                                {scored ? (
                                    lab?.metric_direction === "maximize" ? (
                                        <ArrowUpRight size={14} />
                                    ) : (
                                        <ArrowDownRight size={14} />
                                    )
                                ) : (
                                    <Check size={14} />
                                )}{" "}
                                {scored ? "Score challenge" : "Pass / fail"}
                            </span>
                        </div>
                        {(error || data?.error) && (
                            <div className="error" role="alert">
                                {error || data?.error}
                            </div>
                        )}
                        <div className="table-toolbar">
                            <h3>
                                Standings{" "}
                                <span>{lab?.entries.length ?? 0}</span>
                            </h3>
                            <label className="search">
                                <Search size={15} />
                                <input
                                    aria-label="Search participants"
                                    placeholder="Find a participant…"
                                    value={query}
                                    onChange={(event) =>
                                        setQuery(event.target.value)
                                    }
                                />
                            </label>
                        </div>
                        <div className="table-scroll">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Rank</th>
                                        <th>Participant</th>
                                        <th>
                                            {scored
                                                ? `${lab?.primary_metric} ${lab?.metric_direction === "maximize" ? "↑" : "↓"}`
                                                : "Result"}
                                        </th>
                                        <th>Submitted</th>
                                        <th>
                                            <span className="sr-only">Run</span>
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {entries.map((entry) => (
                                        <tr key={entry.github_actor}>
                                            <td>
                                                <span
                                                    className={
                                                        entry.rank === 1
                                                            ? "rank first"
                                                            : "rank"
                                                    }
                                                >
                                                    {String(
                                                        entry.rank,
                                                    ).padStart(2, "0")}
                                                </span>
                                            </td>
                                            <td>
                                                <div className="participant">
                                                    <GitHubAvatar
                                                        key={entry.github_actor}
                                                        username={
                                                            entry.github_actor
                                                        }
                                                    />
                                                    <div>
                                                        <strong>
                                                            {entry.github_actor}
                                                        </strong>
                                                        <small>
                                                            {entry.attempts}{" "}
                                                            {entry.attempts ===
                                                            1
                                                                ? "submission"
                                                                : "submissions"}
                                                        </small>
                                                    </div>
                                                </div>
                                            </td>
                                            <td>
                                                {scored ? (
                                                    <span className="score">
                                                        {entry.score?.toLocaleString(
                                                            undefined,
                                                            {
                                                                maximumSignificantDigits: 7,
                                                            },
                                                        )}
                                                    </span>
                                                ) : (
                                                    <span className="passed">
                                                        <Check size={12} />{" "}
                                                        Passed
                                                    </span>
                                                )}
                                            </td>
                                            <td className="submitted">
                                                {date(entry.submitted_at)}
                                            </td>
                                            <td>
                                                {runLink(entry.run_url) && (
                                                    <a
                                                        className="run-link"
                                                        href={runLink(
                                                            entry.run_url,
                                                        )!}
                                                        target="_blank"
                                                        rel="noreferrer"
                                                        aria-label={`View ${entry.github_actor}'s W&B run`}
                                                    >
                                                        <External size={16} />
                                                    </a>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        {!entries.length && (
                            <div className="empty">
                                <Trophy size={26} />
                                <h3>
                                    {query
                                        ? "No matching participants"
                                        : loading && !data
                                          ? "Loading standings…"
                                          : "The next result could be yours."}
                                </h3>
                                <p>
                                    {query
                                        ? "Try another GitHub username."
                                        : "Completed, qualifying W&B submissions will appear here."}
                                </p>
                            </div>
                        )}
                        <div className="board-footer">
                            <span>
                                <span className="small-dot" /> Results from
                                Weights & Biases
                            </span>
                        </div>
                    </section>
                </div>
                <footer>
                    <span>
                        <span className="brand-name">Cerulean Labs</span>
                        <span className="footer-muted">/ AI for everyone.</span>
                    </span>
                </footer>
            </main>
        </div>
    );
}
