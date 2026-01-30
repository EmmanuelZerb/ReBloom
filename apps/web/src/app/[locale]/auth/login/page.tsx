'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { useRouter } from '@/i18n/navigation';
import { LoginForm } from '@/components/auth/LoginForm';

function LoginContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirect = searchParams.get('redirect') || '/';

  const handleSuccess = () => {
    router.push(redirect);
  };

  const handleSwitchToSignup = () => {
    router.push('/auth/signup');
  };

  return <LoginForm onSuccess={handleSuccess} onSwitchToSignup={handleSwitchToSignup} />;
}

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#FFF8F4] via-white to-[#F3F8F5] px-4">
      <Suspense fallback={<div className="animate-pulse">Loading...</div>}>
        <LoginContent />
      </Suspense>
    </div>
  );
}
