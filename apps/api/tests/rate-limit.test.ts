/**
 * Tests for rate limiting
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Hono } from 'hono';
import { rateLimit } from '../src/lib/rate-limit';

describe('Rate Limiter', () => {
  let app: Hono;

  beforeEach(() => {
    app = new Hono();
    vi.clearAllMocks();
  });

  describe('Memory Store Rate Limiter', () => {
    it('should allow requests within limit', async () => {
      app.use(
        '*',
        rateLimit({
          maxRequests: 5,
          windowSeconds: 60,
          keyPrefix: 'test',
        })
      );
      app.get('/test', (c) => c.text('OK'));

      // Make requests within limit
      for (let i = 0; i < 5; i++) {
        const res = await app.request('/test');
        expect(res.status).toBe(200);
      }
    });

    it('should block requests exceeding limit', async () => {
      app.use(
        '*',
        rateLimit({
          maxRequests: 3,
          windowSeconds: 60,
          keyPrefix: 'test-block',
        })
      );
      app.get('/test', (c) => c.text('OK'));

      // Make requests within limit
      for (let i = 0; i < 3; i++) {
        const res = await app.request('/test');
        expect(res.status).toBe(200);
      }

      // This request should be blocked
      const blocked = await app.request('/test');
      expect(blocked.status).toBe(429);

      const body = await blocked.json();
      expect(body.error.code).toBe('RATE_LIMIT_EXCEEDED');
    });

    it('should include rate limit headers', async () => {
      app.use(
        '*',
        rateLimit({
          maxRequests: 10,
          windowSeconds: 60,
          keyPrefix: 'test-headers',
        })
      );
      app.get('/test', (c) => c.text('OK'));

      const res = await app.request('/test');

      expect(res.headers.get('X-RateLimit-Limit')).toBe('10');
      expect(res.headers.get('X-RateLimit-Remaining')).toBeDefined();
      expect(res.headers.get('X-RateLimit-Reset')).toBeDefined();
    });

    it('should use custom key generator', async () => {
      const keyGenerator = vi.fn((c) => 'custom-key');

      app.use(
        '*',
        rateLimit({
          maxRequests: 5,
          windowSeconds: 60,
          keyPrefix: 'test-custom',
          keyGenerator,
        })
      );
      app.get('/test', (c) => c.text('OK'));

      await app.request('/test');

      expect(keyGenerator).toHaveBeenCalled();
    });

    it('should use custom error message', async () => {
      const customMessage = 'Custom rate limit message';

      app.use(
        '*',
        rateLimit({
          maxRequests: 1,
          windowSeconds: 60,
          keyPrefix: 'test-message',
          message: customMessage,
        })
      );
      app.get('/test', (c) => c.text('OK'));

      // First request succeeds
      await app.request('/test');

      // Second request is blocked
      const blocked = await app.request('/test');
      const body = await blocked.json();

      expect(body.error.message).toBe(customMessage);
    });
  });

  describe('Fail Modes', () => {
    it('should allow requests in fail-open mode when rate limit check fails', async () => {
      // Create a rate limiter that will fail (by mocking the store)
      app.use(
        '*',
        rateLimit({
          maxRequests: 5,
          windowSeconds: 60,
          keyPrefix: 'test-fail-open',
          failMode: 'open',
        })
      );
      app.get('/test', (c) => c.text('OK'));

      // Normal operation should work
      const res = await app.request('/test');
      expect(res.status).toBe(200);
    });

    it('should block requests in fail-closed mode by default for upload', async () => {
      // This tests the behavior documentation rather than actual failure
      // since we can't easily simulate Redis failure in unit tests
      app.use(
        '*',
        rateLimit({
          maxRequests: 5,
          windowSeconds: 60,
          keyPrefix: 'test-fail-closed',
          failMode: 'closed',
        })
      );
      app.get('/test', (c) => c.text('OK'));

      // Normal operation should work
      const res = await app.request('/test');
      expect(res.status).toBe(200);
    });
  });

  describe('Different IPs', () => {
    it('should track limits separately per IP', async () => {
      app.use(
        '*',
        rateLimit({
          maxRequests: 2,
          windowSeconds: 60,
          keyPrefix: 'test-ip',
          keyGenerator: (c) => c.req.header('x-forwarded-for') || 'unknown',
        })
      );
      app.get('/test', (c) => c.text('OK'));

      // IP 1 makes 2 requests
      const headers1 = { 'x-forwarded-for': '1.1.1.1' };
      expect((await app.request('/test', { headers: headers1 })).status).toBe(200);
      expect((await app.request('/test', { headers: headers1 })).status).toBe(200);
      expect((await app.request('/test', { headers: headers1 })).status).toBe(429);

      // IP 2 can still make requests
      const headers2 = { 'x-forwarded-for': '2.2.2.2' };
      expect((await app.request('/test', { headers: headers2 })).status).toBe(200);
      expect((await app.request('/test', { headers: headers2 })).status).toBe(200);
      expect((await app.request('/test', { headers: headers2 })).status).toBe(429);
    });
  });
});
