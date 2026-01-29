/**
 * Test setup file
 * Runs before all tests
 */

import { vi } from 'vitest';

// Mock environment variables
process.env.NODE_ENV = 'test';
process.env.REPLICATE_API_TOKEN = 'test-token';
process.env.REDIS_URL = 'redis://localhost:6379';
process.env.STORAGE_PROVIDER = 'local';
process.env.STORAGE_LOCAL_PATH = './test-uploads';
process.env.WEBHOOK_SECRET = 'test-webhook-secret';

// Mock Redis
vi.mock('ioredis', () => {
  const Redis = vi.fn(() => ({
    on: vi.fn(),
    get: vi.fn().mockResolvedValue(null),
    set: vi.fn().mockResolvedValue('OK'),
    setex: vi.fn().mockResolvedValue('OK'),
    del: vi.fn().mockResolvedValue(1),
    keys: vi.fn().mockResolvedValue([]),
    incr: vi.fn().mockResolvedValue(1),
    ttl: vi.fn().mockResolvedValue(60),
    expire: vi.fn().mockResolvedValue(1),
    multi: vi.fn(() => ({
      incr: vi.fn().mockReturnThis(),
      ttl: vi.fn().mockReturnThis(),
      exec: vi.fn().mockResolvedValue([[null, 1], [null, 60]]),
    })),
    quit: vi.fn().mockResolvedValue('OK'),
  }));

  return { Redis, default: Redis };
});

// Clean up after tests
afterAll(() => {
  vi.clearAllMocks();
});
