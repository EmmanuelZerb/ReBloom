'use client';

import { Component, ErrorInfo, ReactNode } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Error Boundary Component
 * Catches JavaScript errors anywhere in the child component tree
 * and displays a fallback UI instead of crashing the whole app
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error to console in development
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    // Call optional error handler
    this.props.onError?.(error, errorInfo);
    
    // In production, you might want to log to an error reporting service
    // e.g., Sentry, LogRocket, etc.
  }

  handleRetry = (): void => {
    this.setState({ hasError: false, error: null });
  };

  handleGoHome = (): void => {
    window.location.href = '/';
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return <DefaultErrorFallback error={this.state.error} onRetry={this.handleRetry} onGoHome={this.handleGoHome} />;
    }

    return this.props.children;
  }
}

// ============================================
// Default Error Fallback UI
// ============================================

interface DefaultErrorFallbackProps {
  error: Error | null;
  onRetry: () => void;
  onGoHome: () => void;
}

function DefaultErrorFallback({ error, onRetry, onGoHome }: DefaultErrorFallbackProps) {
  return (
    <div className="min-h-[400px] flex items-center justify-center p-8">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-md w-full"
      >
        <div className="card-crystal p-8 text-center space-y-6">
          {/* Icon */}
          <div className="relative mx-auto w-16 h-16">
            <div className="absolute inset-0 bg-red-500/20 rounded-2xl blur-xl" />
            <div className="relative p-4 rounded-2xl bg-red-500/10 border border-red-500/20">
              <AlertTriangle className="w-8 h-8 text-red-400" />
            </div>
          </div>

          {/* Message */}
          <div className="space-y-2">
            <h3 className="text-xl font-semibold text-white">
              Something went wrong
            </h3>
            <p className="text-sm text-white/50">
              {error?.message || 'An unexpected error occurred. Please try again.'}
            </p>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-center gap-3">
            <motion.button
              onClick={onRetry}
              className="btn-prism flex items-center gap-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <RefreshCw className="w-4 h-4" />
              <span>Try Again</span>
            </motion.button>

            <motion.button
              onClick={onGoHome}
              className="btn-ghost flex items-center gap-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Home className="w-4 h-4" />
              <span>Go Home</span>
            </motion.button>
          </div>

          {/* Debug info in dev */}
          {process.env.NODE_ENV === 'development' && error && (
            <details className="text-left">
              <summary className="text-xs text-white/30 cursor-pointer hover:text-white/50">
                Error details (dev only)
              </summary>
              <pre className="mt-2 p-3 bg-white/5 rounded-lg text-xs text-red-300 overflow-auto">
                {error.stack}
              </pre>
            </details>
          )}
        </div>
      </motion.div>
    </div>
  );
}

// ============================================
// Hook for programmatic error handling
// ============================================

export function useErrorHandler() {
  const handleError = (error: Error) => {
    // In a real app, you might want to:
    // 1. Log to error tracking service
    // 2. Show a toast notification
    // 3. Update global state
    console.error('Error handled:', error);
  };

  return { handleError };
}

