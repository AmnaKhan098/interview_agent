"use client";
import React, { useState, useEffect } from "react";
import { Briefcase, Code, Users, MessageCircle } from "lucide-react";
import { useRouter } from "next/navigation";

// Define the stats structure
type StatsItem = {
  label: string;
  value: number;
  icon: React.ElementType;
  color: string;
};

// Initial stats with zero values
const initialStats: StatsItem[] = [
  {
    label: "Interviews Completed",
    value: 0,
    icon: Briefcase,
    color: "text-blue-500",
  },
  {
    label: "Coding Challenges",
    value: 0,
    icon: Code,
    color: "text-purple-500",
  },
  {
    label: "Behavioral Sessions",
    value: 0,
    icon: Users,
    color: "text-pink-500",
  },
  {
    label: "Community Posts",
    value: 0,
    icon: MessageCircle,
    color: "text-green-500",
  },
];

export default function DashboardPage() {
  const router = useRouter();
  const [stats, setStats] = useState<StatsItem[]>(initialStats);
  const [recentInterviews, setRecentInterviews] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDashboardStats = async () => {
      try {
        if (loading) {
          setError(null);
        }

        setLoading(true);
        const response = await fetch('/api/dashboard');
        
        if (response.status === 401) {
          router.push('/login');
          return;
        }
        
        if (!response.ok) {
          throw new Error('Failed to fetch dashboard statistics');
        }
        
        const data = await response.json();
        
        // Update stats with real data
        setStats([
          {
            label: "Interviews Completed",
            value: data.interviewsCompleted,
            icon: Briefcase,
            color: "text-blue-500",
          },
          {
            label: "Coding Challenges",
            value: data.codingChallenges || 0,
            icon: Code,
            color: "text-purple-500",
          },
          {
            label: "Behavioral Sessions",
            value: data.behavioralSessions || 0,
            icon: Users,
            color: "text-pink-500",
          },
          // {
          //   label: "Community Posts",
          //   value: data.communityPosts || 0,
          //   icon: MessageCircle,
          //   color: "text-green-500",
          // },
        ]);

        // Store recent interviews to pass to the Recent Activity section
        setRecentInterviews(data.recentInterviews || []);
      } catch (err) {
        console.error('Error fetching dashboard stats:', err);
        setError('Failed to load dashboard statistics');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardStats();
  }, []);
  

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-100 py-12 px-4 flex flex-col items-center">
      <div className="w-full max-w-7xl relative">
        <div className="flex justify-end items-center mb-10">
          {/* Avatar dropdown removed, now handled globally in layout */}
        </div>
        <h1 className="text-4xl font-extrabold text-gray-900 mb-10 tracking-tight">
          Dashboard
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
          {stats.map((stat) => (
            <div
              key={stat.label}
              className="bg-white rounded-2xl shadow-xl p-8 flex flex-col items-center border border-gray-100 hover:shadow-2xl transition-all group relative overflow-hidden"
            >
              <div
                className={`absolute -top-6 -right-6 opacity-10 text-8xl pointer-events-none select-none ${stat.color}`}
              >
                <stat.icon className="w-24 h-24" />
              </div>
              <div className={`mb-3 z-10 ${stat.color}`}>
                <stat.icon className="w-8 h-8" />
              </div>
              <div className="text-3xl font-bold text-gray-800 mb-1 z-10">
                {stat.value}
              </div>
              <div className="text-gray-500 text-base font-medium text-center z-10">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
          <div className="lg:col-span-2 bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <h2 className="text-2xl font-semibold mb-6 text-gray-800">
              Recent Activity
            </h2>
            <ul className="divide-y divide-gray-100">
              {loading ? (
                <li className="py-4 text-gray-500">Loading recent interviews...</li>
              ) : error ? (
                <li className="py-4 text-red-500">Error: {error}</li>
              ) : recentInterviews.length === 0 ? (
                <li className="py-4 text-gray-500">No recent interviews.</li>
              ) : (
                recentInterviews.map((activity: any) => (
                  <li key={activity.id} className="py-4 flex items-center justify-between">
                    <a href={`/feedback?interviewId=${activity.id}`} className="text-gray-700 font-medium hover:underline">
                      {activity.jobTitle}
                    </a>
                    <span className="text-xs text-gray-400">
                      {new Date(activity.date).toLocaleDateString()}
                    </span>
                  </li>
                ))
              )}
            </ul>
          </div>
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 flex flex-col justify-between">
            <h2 className="text-2xl font-semibold mb-6 text-gray-800">
              Quick Links
            </h2>
            <ul className="space-y-4">
              <li>
                <a
                  href="/practice"
                  className="flex items-center gap-2 text-blue-600 hover:underline font-medium transition-colors hover:text-blue-800"
                >
                  <Code className="w-5 h-5" /> Practice Coding
                </a>
              </li>
              <li>
                <a
                  href="/practice/mock"
                  className="flex items-center gap-2 text-blue-600 hover:underline font-medium transition-colors hover:text-blue-800"
                >
                  <Briefcase className="w-5 h-5" /> Mock Interviews
                </a>
              </li>
              <li>
                <a
                  href="/community/discussions"
                  className="flex items-center gap-2 text-blue-600 hover:underline font-medium transition-colors hover:text-blue-800"
                >
                  <MessageCircle className="w-5 h-5" /> Community Discussions
                </a>
              </li>
            </ul>
          </div>
        </div>

      </div>
    </div>
  );
}
