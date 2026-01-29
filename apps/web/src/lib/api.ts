/**
 * API Client pour ReBloom
 */

import type {
  UploadResponse,
  JobStatusResponse,
  DownloadResponse,
  ErrorResponse,
  EnhanceOptions,
} from '@rebloom/shared';
import { API_ROUTES } from '@rebloom/shared';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

// Token getter - will be set by the auth context
let getAuthToken: (() => Promise<string | null>) | null = null;

/**
 * Set the auth token getter function
 * Called by AuthContext when initialized
 */
export function setAuthTokenGetter(getter: () => Promise<string | null>) {
  getAuthToken = getter;
}

/**
 * Get authorization headers if authenticated
 */
async function getAuthHeaders(): Promise<Record<string, string>> {
  if (!getAuthToken) return {};

  const token = await getAuthToken();
  if (!token) return {};

  return { Authorization: `Bearer ${token}` };
}

class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  const data = await response.json();

  if (!response.ok || data.success === false) {
    const error = (data as ErrorResponse).error;
    throw new ApiError(error.code, error.message, error.details);
  }

  return data as T;
}

/**
 * Upload une image pour traitement
 */
export async function uploadImage(
  file: File,
  options?: Partial<EnhanceOptions>
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  if (options) {
    formData.append('options', JSON.stringify(options));
  }

  const authHeaders = await getAuthHeaders();

  const response = await fetch(`${API_BASE}${API_ROUTES.upload}`, {
    method: 'POST',
    headers: authHeaders,
    body: formData,
  });

  return handleResponse<UploadResponse>(response);
}

/**
 * Récupère le statut d'un job
 */
export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const authHeaders = await getAuthHeaders();
  const response = await fetch(`${API_BASE}${API_ROUTES.jobStatus(jobId)}`, {
    headers: authHeaders,
  });
  return handleResponse<JobStatusResponse>(response);
}

/**
 * Récupère le résultat d'un job terminé
 */
export async function getJobResult(jobId: string): Promise<DownloadResponse> {
  const authHeaders = await getAuthHeaders();
  const response = await fetch(`${API_BASE}${API_ROUTES.jobResult(jobId)}`, {
    headers: authHeaders,
  });
  return handleResponse<DownloadResponse>(response);
}

/**
 * Génère l'URL de téléchargement
 * Note: In production, use getJobResult() to get a signed URL instead
 */
export function getDownloadUrl(jobId: string): string {
  return `${API_BASE}${API_ROUTES.download(jobId)}`;
}

/**
 * Get signed download URL from job result
 * This is the preferred method in production as it returns a secure signed URL
 */
export async function getSignedDownloadUrl(jobId: string): Promise<string> {
  const result = await getJobResult(jobId);
  return `${API_BASE}${result.downloadUrl}`;
}

/**
 * Polling pour le statut d'un job
 */
export async function pollJobStatus(
  jobId: string,
  onProgress: (job: JobStatusResponse['job']) => void,
  options: { intervalMs?: number; maxAttempts?: number } = {}
): Promise<JobStatusResponse['job']> {
  const { intervalMs = 2000, maxAttempts = 150 } = options;
  let attempts = 0;

  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const response = await getJobStatus(jobId);
        onProgress(response.job);

        if (response.job.status === 'completed') {
          resolve(response.job);
          return;
        }

        if (response.job.status === 'failed') {
          reject(new ApiError('JOB_FAILED', response.job.error?.message || 'Job failed'));
          return;
        }

        attempts++;
        if (attempts >= maxAttempts) {
          reject(new ApiError('TIMEOUT', 'Job polling timeout'));
          return;
        }

        setTimeout(poll, intervalMs);
      } catch (error) {
        reject(error);
      }
    };

    poll();
  });
}

export { ApiError };
