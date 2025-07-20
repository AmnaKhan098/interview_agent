import { NextResponse } from 'next/server';
import { openDb } from '@/src/lib/db';

import { cookies } from 'next/headers';
import { getAuthUser } from '@/src/lib/auth';
export async function GET(request: Request) {
  try {
    const user = await getAuthUser(cookies());

    if (!user) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const userId = user.id;

    // If you don't need user-specific data for the dashboard, you can remove the userId filter
    // and fetch all completed interviews.


    const db = await openDb();

    // Fetch interviews completed by the user
    const interviewsCompletedResult = await db.get(
      `SELECT COUNT(*) as count FROM interviews WHERE user_id = ? AND status = 'Completed'`,
      userId
    );
    const interviewsCompleted = interviewsCompletedResult ? interviewsCompletedResult.count : 0;

    // Fetch recent interview activities (e.g., last 3 completed interviews)
    const recentActivities = await db.all(
      `SELECT id, job_title, created_at FROM interviews WHERE user_id = ? AND status = 'Completed' ORDER BY created_at DESC LIMIT 3`,
      userId
    );

    return NextResponse.json({
      interviewsCompleted: interviewsCompleted,
      codingChallenges: 0, // Placeholder for now
      behavioralSessions: 0, // Placeholder for now
      communityPosts: 0, // Placeholder for now
      recentInterviews: recentActivities.map((activity: any) => ({
        id: activity.id,
        jobTitle: activity.job_title,
        date: activity.created_at,
      })),
    });
  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    return NextResponse.json({ error: 'Failed to fetch dashboard data' }, { status: 500 });
  }
}