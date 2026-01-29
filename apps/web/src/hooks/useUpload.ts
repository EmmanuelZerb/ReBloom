'use client';

import { useState, useCallback, useEffect } from 'react';
import { uploadImage, pollJobStatus, getSignedDownloadUrl, ApiError } from '@/lib/api';
import type { Job, EnhanceOptions } from '@rebloom/shared';

export type UploadState =
  | { status: 'idle' }
  | { status: 'uploading'; progress: number }
  | { status: 'processing'; job: Job }
  | { status: 'completed'; job: Job; downloadUrl: string }
  | { status: 'error'; error: string };

export interface UseUploadResult {
  state: UploadState;
  upload: (file: File, options?: Partial<EnhanceOptions>) => Promise<void>;
  reset: () => void;
  previewUrl: string | null;
}

/**
 * Custom hook for handling image upload and processing workflow
 * Manages the complete lifecycle: upload → process → complete/error
 */
export function useUpload(): UseUploadResult {
  const [state, setState] = useState<UploadState>({ status: 'idle' });
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Cleanup preview URL on unmount
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const upload = useCallback(async (file: File, options?: Partial<EnhanceOptions>) => {
    try {
      // Create local preview
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);

      // Start upload
      setState({ status: 'uploading', progress: 0 });

      const uploadResponse = await uploadImage(file, options);

      // Start polling for job status
      setState({
        status: 'processing',
        job: {
          id: uploadResponse.jobId,
          status: 'pending',
          progress: 10,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          originalImage: uploadResponse.originalImage,
          metadata: {
            modelUsed: 'Real-ESRGAN',
            provider: 'replicate',
          },
        },
      });

      const completedJob = await pollJobStatus(
        uploadResponse.jobId,
        (job) => {
          setState({ status: 'processing', job });
        }
      );

      // Get signed download URL
      const downloadUrl = await getSignedDownloadUrl(completedJob.id);

      // Success
      setState({
        status: 'completed',
        job: completedJob,
        downloadUrl,
      });
    } catch (error) {
      // Extract safe error message
      const message =
        error instanceof ApiError
          ? error.message
          : error instanceof Error
            ? error.message
            : 'An unexpected error occurred';

      setState({ status: 'error', error: message });
    }
  }, []);

  const reset = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setState({ status: 'idle' });
  }, [previewUrl]);

  return { state, upload, reset, previewUrl };
}
