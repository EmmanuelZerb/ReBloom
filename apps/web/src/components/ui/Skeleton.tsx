'use client';

import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
  variant?: 'default' | 'circular' | 'text' | 'card';
  animate?: boolean;
}

/**
 * Skeleton loading component
 * Provides visual feedback while content is loading
 */
export function Skeleton({ 
  className, 
  variant = 'default',
  animate = true 
}: SkeletonProps) {
  const baseClasses = cn(
    'bg-white/5 rounded-lg',
    animate && 'animate-pulse',
    className
  );

  const variantClasses = {
    default: '',
    circular: 'rounded-full',
    text: 'h-4 rounded',
    card: 'rounded-2xl',
  };

  return (
    <div 
      className={cn(baseClasses, variantClasses[variant])}
      role="status"
      aria-label="Loading..."
    />
  );
}

// ============================================
// Preset Skeleton Layouts
// ============================================

export function DropZoneSkeleton() {
  return (
    <div className="card-crystal p-8 space-y-6 min-h-[380px] flex flex-col items-center justify-center">
      <Skeleton className="w-20 h-20" variant="circular" />
      <div className="space-y-2 text-center">
        <Skeleton className="w-48 h-6 mx-auto" />
        <Skeleton className="w-32 h-4 mx-auto" />
      </div>
      <div className="flex gap-3">
        <Skeleton className="w-16 h-8" />
        <Skeleton className="w-16 h-8" />
        <Skeleton className="w-16 h-8" />
      </div>
    </div>
  );
}

export function ImageCompareSkeleton() {
  return (
    <div className="card-crystal p-4 space-y-4">
      <Skeleton className="w-full aspect-video" variant="card" />
      <div className="flex justify-between">
        <Skeleton className="w-24 h-8" />
        <Skeleton className="w-24 h-8" />
      </div>
    </div>
  );
}

export function ProcessingStatusSkeleton() {
  return (
    <div className="card-crystal p-8 space-y-6">
      <div className="flex items-center gap-4">
        <Skeleton className="w-12 h-12" variant="circular" />
        <div className="space-y-2 flex-1">
          <Skeleton className="w-32 h-5" />
          <Skeleton className="w-48 h-4" />
        </div>
      </div>
      <div className="space-y-3">
        <div className="flex justify-between">
          <Skeleton className="w-16 h-4" />
          <Skeleton className="w-12 h-4" />
        </div>
        <Skeleton className="w-full h-2" />
      </div>
      <div className="flex justify-between">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="flex flex-col items-center gap-2">
            <Skeleton className="w-3 h-3" variant="circular" />
            <Skeleton className="w-12 h-3" />
          </div>
        ))}
      </div>
    </div>
  );
}

export function FeatureCardSkeleton() {
  return (
    <div className="card-crystal p-6 space-y-4">
      <div className="flex items-start gap-4">
        <Skeleton className="w-12 h-12 shrink-0" />
        <div className="space-y-2 flex-1">
          <Skeleton className="w-3/4 h-5" />
          <Skeleton className="w-full h-4" />
          <Skeleton className="w-2/3 h-4" />
        </div>
      </div>
    </div>
  );
}

