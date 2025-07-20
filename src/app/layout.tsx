"use client";

import Sidebar from "../components/layout/Sidebar";
import { Footer } from "../components/layout/Footer";
import "./globals.css";
import Script from "next/script";
import { useState } from "react";
import AvatarDropdown from "../components/layout/AvatarDropdown";
import { AuthProvider } from "../lib/contexts/AuthContext";

import {
  ClerkProvider,
  SignInButton,
  SignUpButton,
  SignedIn,
  SignedOut,
  UserButton,
} from '@clerk/nextjs'

// Sidebar is now the main navigation. Navbar removed.

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <ClerkProvider>
      <html lang="en">
        <head>
          <link rel="preconnect" href="https://fonts.googleapis.com" />
          <link
            rel="preconnect"
            href="https://fonts.gstatic.com"
            crossOrigin="anonymous"
          />
          <link
            href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
            rel="stylesheet"
          />
        </head>
        <body className="min-h-screen w-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 overflow-x-hidden">
          <AuthProvider>
            {/* Sidebar */}
            <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
            {/* Hamburger for mobile */}
            <button
              className="fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow-lg"
              onClick={() => setSidebarOpen(true)}
            >
              <svg
                className="w-6 h-6 text-gray-800"
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            </button>
            {/* Avatar at top right */}
            <div className="fixed top-6 right-20 z-50 flex items-center">
              <SignedOut>
                <div className="bg-white text-black border border-gray-300 px-4 py-2 rounded hover:bg-gray-100">
                  <SignInButton />
                </div>
              </SignedOut>
              <SignedIn>
                <UserButton
                  appearance={{
                    elements: {
                      userButtonAvatarBox: {
                        width: '48px',
                        height: '48px',
                      },
                    },
                  }}
                />

              </SignedIn>
            </div>
            {/* Main content */}
            <div className="flex flex-col min-h-screen">
              <main className="w-full flex-grow">{children}</main>
              <Footer />
            </div>
          </AuthProvider>
          <Script id="intersection-observer">
            {`document.addEventListener('DOMContentLoaded', function() {
            const observerOptions = {
              root: null,
              rootMargin: '0px',
              threshold: 0.1
            };
            const observer = new IntersectionObserver((entries) => {
              entries.forEach(entry => {
                if (entry.isIntersecting) {
                  entry.target.classList.add('appear');
                  observer.unobserve(entry.target);
                }
              });
            }, observerOptions);
            const animatedElements = document.querySelectorAll('.fade-in, .from-bottom, .from-left, .from-right');
            animatedElements.forEach(el => observer.observe(el));
            // Parallax effect
            const parallaxElements = document.querySelectorAll('.parallax');
            window.addEventListener('scroll', () => {
              parallaxElements.forEach(element => {
                const bg = element.querySelector('.parallax-bg');
                if (bg) {
                  const scrollPosition = window.pageYOffset;
                  const speed = 0.5;
                  bg.style.transform = 'translateY(' + scrollPosition * speed + 'px)';
                }
              });
            });
          });`}
          </Script>
        </body>
      </html>
    </ClerkProvider>
  );
}
