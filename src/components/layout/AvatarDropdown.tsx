'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/src/lib/contexts/AuthContext';

export default function AvatarDropdown() {
  const router = useRouter();
  const { user, isAuthenticated, logout } = useAuth();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const avatarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        avatarRef.current &&
        !avatarRef.current.contains(event.target as Node)
      ) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = async () => {
    try {
      await logout();
      router.push('/');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  if (!isAuthenticated) {
    // return (
    //   <div className="fixed top-4 right-6 z-50">
    //     <button
    //       onClick={() => router.push('/login')}
    //       className="px-4 py-2 bg-white rounded-lg shadow-md text-blue-600 font-medium hover:bg-blue-50 transition-colors"
    //     >
    //       Sign In
    //     </button>
    //   </div>
    // );
  }

  // return (
  //   <div className="fixed top-4 right-6 z-50" ref={avatarRef}>
  //     <img
  //       src={user?.avatar || `https://ui-avatars.com/api/?name=${encodeURIComponent(user?.name || 'User')}`}
  //       alt="avatar"
  //       className="w-12 h-12 rounded-full border-2 border-purple-400 shadow cursor-pointer object-cover"
  //       onClick={() => setDropdownOpen((v) => !v)}
  //     />
  //     {dropdownOpen && (
  //       <div className="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-lg py-2 z-50 border border-gray-100 animate-fade-in">
  //         <div className="px-4 py-2 border-b border-gray-100">
  //           <p className="text-sm font-semibold text-gray-900 truncate">{user?.name}</p>
  //           <p className="text-xs text-gray-500 truncate">{user?.email}</p>
  //         </div>
  //         <button
  //           className="w-full text-left px-4 py-2 hover:bg-gray-100 text-gray-800 font-medium"
  //           onClick={() => {
  //             setDropdownOpen(false);
  //             router.push('/profile');
  //           }}
  //         >
  //           Profile
  //         </button>
  //         <button
  //           className="w-full text-left px-4 py-2 hover:bg-gray-100 text-gray-800 font-medium"
  //           onClick={() => {
  //             setDropdownOpen(false);
  //             router.push('/dashboard');
  //           }}
  //         >
  //           Dashboard
  //         </button>
  //         <button
  //           className="w-full text-left px-4 py-2 hover:bg-gray-100 text-red-500 font-medium"
  //           onClick={() => {
  //             setDropdownOpen(false);
  //             handleLogout();
  //           }}
  //         >
  //           Logout
  //         </button>
  //       </div>
  //     )}
  //   </div>
  // );
}
