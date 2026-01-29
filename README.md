# 🌸 ReBloom

> Transform blurry photos into stunning clarity with AI-powered image enhancement.

ReBloom is a production-ready SaaS for image deblurring and enhancement, powered by Real-ESRGAN. Built with a modern stack: Next.js 14, Hono.js, BullMQ, and Replicate.

![ReBloom Demo](docs/demo.gif)

## Features

- **AI-Powered Enhancement** - Real-ESRGAN for state-of-the-art image restoration
- **4x Upscaling** - Increase resolution while adding detail
- **Async Processing** - Queue-based architecture with real-time progress updates
- **Modern UI** - Drag & drop upload, before/after comparison slider
- **Production Ready** - Error handling, logging, rate limiting
- **Fine-tuning Ready** - Complete setup for training your own model

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 14, React 18, Tailwind CSS, Framer Motion |
| **Backend** | Hono.js, BullMQ, Redis |
| **AI** | Replicate API (Real-ESRGAN) |
| **Monorepo** | Turborepo, TypeScript |

## Quick Start

### Prerequisites

- Node.js 20+
- Docker (for Redis)
- Replicate API token ([get one here](https://replicate.com))

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/rebloom.git
cd rebloom

# Install dependencies
npm install

# Start Redis
docker-compose up -d redis

# Copy environment file
cp .env.example .env

# Add your Replicate API token to .env
# REPLICATE_API_TOKEN=r8_your_token_here

# Start development servers
npm run dev
```

### Access the app

- **Frontend**: http://localhost:3000
- **API**: http://localhost:3001
- **Redis UI** (optional): http://localhost:8081

## Project Structure

```
rebloom/
├── apps/
│   ├── web/                 # Next.js frontend
│   │   ├── src/
│   │   │   ├── app/         # Pages and layouts
│   │   │   ├── components/  # UI components
│   │   │   ├── hooks/       # Custom hooks
│   │   │   └── lib/         # Utilities
│   │   └── package.json
│   │
│   └── api/                 # Hono.js backend
│       ├── src/
│       │   ├── routes/      # API endpoints
│       │   ├── services/    # Business logic
│       │   ├── workers/     # Queue processors
│       │   └── lib/         # Utilities
│       └── package.json
│
├── packages/
│   └── shared/              # Shared types and constants
│
├── training/                # Fine-tuning setup
│   ├── scripts/             # Dataset generation & training
│   ├── configs/             # Training configuration
│   └── docs/                # Fine-tuning guide
│
├── docs/                    # Documentation
├── docker-compose.yml       # Local services
└── turbo.json               # Turborepo config
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload image for processing |
| `GET` | `/api/jobs/:id` | Get job status |
| `GET` | `/api/jobs/:id/result` | Get processed image URL |
| `GET` | `/api/download/:id` | Download processed image |

### Example: Upload Image

```bash
curl -X POST http://localhost:3001/api/upload \
  -F "file=@photo.jpg" \
  -F 'options={"scaleFactor": 4, "faceEnhance": false}'
```

Response:
```json
{
  "success": true,
  "jobId": "abc123",
  "originalImage": {
    "width": 1024,
    "height": 768
  },
  "estimatedTimeSeconds": 15
}
```

## Configuration

All configuration is done via environment variables. See [.env.example](.env.example) for all options.

Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `REPLICATE_API_TOKEN` | Your Replicate API key | Required |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `MAX_FILE_SIZE_MB` | Max upload size | `20` |
| `STORAGE_PROVIDER` | `local` or `s3` | `local` |

## Fine-Tuning Your Own Model

ReBloom is designed to be extended with your own fine-tuned model. See the complete guide:

📖 [**Fine-Tuning Guide**](docs/FINE_TUNING_GUIDE.md)

Quick overview:
1. Prepare your dataset with sharp/blur pairs
2. Use provided scripts to generate synthetic blur
3. Fine-tune Real-ESRGAN with your data
4. Deploy to Replicate
5. Update ReBloom config to use your model

## Scripts

```bash
# Development
npm run dev          # Start all services
npm run web:dev      # Frontend only
npm run api:dev      # Backend only

# Database
npm run db:setup     # Start Redis

# Build
npm run build        # Build all packages
npm run lint         # Lint all packages

# Training (in training/ directory)
python scripts/generate_blur.py   # Generate blur/sharp pairs
python scripts/prepare_dataset.py # Prepare for training
python scripts/train.py           # Start fine-tuning
```

## Deployment

### Vercel (Frontend)

```bash
cd apps/web
vercel
```

### Railway/Render (Backend)

The API can be deployed to any Node.js hosting. Make sure to:
1. Set all environment variables
2. Provision a Redis instance
3. Configure storage (S3 recommended for production)

### Docker

```dockerfile
# Coming soon
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - The AI model powering ReBloom
- [Replicate](https://replicate.com) - Easy model deployment
- [BasicSR](https://github.com/XPixelGroup/BasicSR) - Training framework

---

Built with 💜 using AI-powered development
