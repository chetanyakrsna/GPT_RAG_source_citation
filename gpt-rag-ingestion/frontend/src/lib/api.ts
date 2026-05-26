export type SortOrder = "asc" | "desc";

export interface QueryParams {
  page: number;
  pageSize: number;
  search: string;
  sortField: string;
  sortOrder: SortOrder;
  indexerType?: string;
  blocked?: boolean;
}

export interface JobRun {
  _blobName?: string;
  indexerType?: string;
  runId?: string;
  status?: string;
  runStartedAt?: string;
  runFinishedAt?: string;
  candidates?: number;
  skippedNoChange?: number;
  skippedBlocked?: number;
  indexedItems?: number;
  indexParentsPurged?: number;
  failed?: number;
  [key: string]: unknown;
}

export interface FileLog {
  _blobName?: string;
  indexerType?: string;
  fileName?: string;
  blob?: string;
  parent_id?: string;
  status?: string;
  blocked?: boolean;
  processingAttempts?: number;
  startedAt?: string;
  finishedAt?: string;
  runHistory?: unknown[];
  [key: string]: unknown;
}

export interface PagedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  indexerTypes?: string[];
}

const API_BASE = "/api";

function buildQuery(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === "") continue;
    search.set(key, String(value));
  }
  const query = search.toString();
  return query ? `?${query}` : "";
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!res.ok) {
    throw new Error(`Request failed (${res.status}) for ${path}`);
  }

  return (await res.json()) as T;
}

export function formatUtc(value?: string | null): string {
  if (!value) return "-";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString("en-US", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: "UTC",
  });
}

export async function fetchVersion(): Promise<string> {
  const data = await fetchJson<{ version?: string }>("/version");
  return data.version ?? "";
}

export async function fetchJobs(
  params: Omit<QueryParams, "blocked">,
  signal?: AbortSignal
): Promise<PagedResponse<JobRun>> {
  const query = buildQuery({
    page: params.page,
    pageSize: params.pageSize,
    search: params.search,
    sortField: params.sortField,
    sortOrder: params.sortOrder,
    indexerType: params.indexerType,
  });
  return fetchJson<PagedResponse<JobRun>>(`/jobs${query}`, { method: "GET", signal });
}

export async function fetchFiles(params: QueryParams, signal?: AbortSignal): Promise<PagedResponse<FileLog>> {
  const query = buildQuery({
    page: params.page,
    pageSize: params.pageSize,
    search: params.search,
    sortField: params.sortField,
    sortOrder: params.sortOrder,
    blocked: params.blocked,
    indexerType: params.indexerType,
  });
  return fetchJson<PagedResponse<FileLog>>(`/files${query}`, { method: "GET", signal });
}

export async function unblockFile(blobName: string): Promise<{ status: string; blobName: string }> {
  const query = buildQuery({ blobName });
  return fetchJson<{ status: string; blobName: string }>(`/files/unblock${query}`, { method: "POST" });
}
