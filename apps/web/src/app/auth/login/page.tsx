'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { LoginForm } from '@/components/auth/LoginForm';

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirect = searchParams.get('redirect') || '/';

  const handleSuccess = () => {
    router.push(redirect);
  };

  const handleSwitchToSignup = () => {
    router.push('/auth/signup');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#FFF8F4] via-white to-[#F3F8F5] px-4">
      <LoginForm onSuccess={handleSuccess} onSwitchToSignup={handleSwitchToSignup} />
    </div>
  );
}
