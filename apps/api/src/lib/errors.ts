/**
 * Custom error classes et gestion d'erreurs
 */

import type { ErrorCode } from '@rebloom/shared';
import { ERROR_CODES } from '@rebloom/shared';

export class AppError extends Error {
  constructor(
    public readonly code: ErrorCode,
    message: string,
    public readonly statusCode: number = 500,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'AppError';
  }

  toJSON() {
    return {
      success: false as const,
      error: {
        code: this.code,
        message: this.message,
        ...(this.details && { details: this.details }),
      },
    };
  }
}

// Erreurs spécifiques

export class ValidationError extends AppError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(ERROR_CODES.INVALID_FILE_TYPE, message, 400, details);
    this.name = 'ValidationError';
  }
}

export class FileTooLargeError extends AppError {
  constructor(maxSize: number, actualSize: number) {
    super(
      ERROR_CODES.FILE_TOO_LARGE,
      `File size ${formatBytes(actualSize)} exceeds maximum ${formatBytes(maxSize)}`,
      413,
      { maxSize, actualSize }
    );
    this.name = 'FileTooLargeError';
  }
}

export class InvalidFileTypeError extends AppError {
  constructor(mimeType: string, allowed: string[]) {
    super(
      ERROR_CODES.INVALID_FILE_TYPE,
      `File type ${mimeType} is not allowed. Allowed types: ${allowed.join(', ')}`,
      415,
      { mimeType, allowed }
    );
    this.name = 'InvalidFileTypeError';
  }
}

export class JobNotFoundError extends AppError {
  constructor(jobId: string) {
    super(ERROR_CODES.JOB_NOT_FOUND, `Job ${jobId} not found`, 404, { jobId });
    this.name = 'JobNotFoundError';
  }
}

export class ProcessingError extends AppError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(ERROR_CODES.PROCESSING_FAILED, message, 500, details);
    this.name = 'ProcessingError';
  }
}

export class ProviderError extends AppError {
  constructor(provider: string, message: string, details?: Record<string, unknown>) {
    super(ERROR_CODES.PROVIDER_ERROR, `${provider}: ${message}`, 502, { provider, ...details });
    this.name = 'ProviderError';
  }
}

// Helpers

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
