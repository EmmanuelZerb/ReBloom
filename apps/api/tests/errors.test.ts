/**
 * Tests for custom error classes
 */

import { describe, it, expect } from 'vitest';
import {
  AppError,
  ValidationError,
  FileTooLargeError,
  InvalidFileTypeError,
  JobNotFoundError,
  ProcessingError,
  ProviderError,
  UnauthorizedError,
  WebhookSignatureError,
  ImageDimensionError,
} from '../src/lib/errors';
import { ERROR_CODES } from '@rebloom/shared';

describe('Error Classes', () => {
  describe('AppError', () => {
    it('should create an error with correct properties', () => {
      const error = new AppError(ERROR_CODES.INTERNAL_ERROR, 'Test error', 500);

      expect(error.code).toBe(ERROR_CODES.INTERNAL_ERROR);
      expect(error.message).toBe('Test error');
      expect(error.statusCode).toBe(500);
      expect(error.name).toBe('AppError');
    });

    it('should serialize to JSON correctly', () => {
      const error = new AppError(ERROR_CODES.INTERNAL_ERROR, 'Test error', 500, {
        extra: 'data',
      });

      const json = error.toJSON();

      expect(json.success).toBe(false);
      expect(json.error.code).toBe(ERROR_CODES.INTERNAL_ERROR);
      expect(json.error.message).toBe('Test error');
      expect(json.error.details).toEqual({ extra: 'data' });
    });
  });

  describe('ValidationError', () => {
    it('should have correct status code 400', () => {
      const error = new ValidationError('Invalid input');

      expect(error.statusCode).toBe(400);
      expect(error.name).toBe('ValidationError');
    });
  });

  describe('FileTooLargeError', () => {
    it('should format size information correctly', () => {
      const error = new FileTooLargeError(20 * 1024 * 1024, 30 * 1024 * 1024);

      expect(error.statusCode).toBe(413);
      expect(error.message).toContain('20');
      expect(error.message).toContain('30');
      expect(error.details?.maxSize).toBe(20 * 1024 * 1024);
      expect(error.details?.actualSize).toBe(30 * 1024 * 1024);
    });
  });

  describe('InvalidFileTypeError', () => {
    it('should list allowed types', () => {
      const error = new InvalidFileTypeError('image/gif', ['image/jpeg', 'image/png']);

      expect(error.statusCode).toBe(415);
      expect(error.message).toContain('image/gif');
      expect(error.message).toContain('image/jpeg');
      expect(error.message).toContain('image/png');
    });
  });

  describe('JobNotFoundError', () => {
    it('should include job ID', () => {
      const error = new JobNotFoundError('job-123');

      expect(error.statusCode).toBe(404);
      expect(error.message).toContain('job-123');
      expect(error.details?.jobId).toBe('job-123');
    });
  });

  describe('ProcessingError', () => {
    it('should have status code 500', () => {
      const error = new ProcessingError('Processing failed');

      expect(error.statusCode).toBe(500);
      expect(error.code).toBe(ERROR_CODES.PROCESSING_FAILED);
    });
  });

  describe('ProviderError', () => {
    it('should include provider name', () => {
      const error = new ProviderError('Replicate', 'API timeout');

      expect(error.statusCode).toBe(502);
      expect(error.message).toContain('Replicate');
      expect(error.message).toContain('API timeout');
    });
  });

  describe('UnauthorizedError', () => {
    it('should have status code 401', () => {
      const error = new UnauthorizedError('Invalid token');

      expect(error.statusCode).toBe(401);
      expect(error.code).toBe(ERROR_CODES.UNAUTHORIZED);
    });
  });

  describe('WebhookSignatureError', () => {
    it('should have status code 401', () => {
      const error = new WebhookSignatureError({ reason: 'Invalid signature' });

      expect(error.statusCode).toBe(401);
      expect(error.details?.reason).toBe('Invalid signature');
    });
  });

  describe('ImageDimensionError', () => {
    it('should detect image too small', () => {
      const error = new ImageDimensionError(32, 32, 64, 4096);

      expect(error.statusCode).toBe(400);
      expect(error.code).toBe(ERROR_CODES.IMAGE_TOO_SMALL);
      expect(error.message).toContain('too small');
    });

    it('should detect image too large', () => {
      const error = new ImageDimensionError(5000, 5000, 64, 4096);

      expect(error.statusCode).toBe(400);
      expect(error.code).toBe(ERROR_CODES.IMAGE_TOO_LARGE);
      expect(error.message).toContain('too large');
    });

    it('should include dimension info', () => {
      const error = new ImageDimensionError(5000, 3000, 64, 4096);

      expect(error.details?.width).toBe(5000);
      expect(error.details?.height).toBe(3000);
      expect(error.details?.minDim).toBe(64);
      expect(error.details?.maxDim).toBe(4096);
    });
  });
});
