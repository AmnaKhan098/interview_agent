import { NextRequest, NextResponse } from 'next/server';
import { openDb } from '@/src/lib/db';
import { cookies } from 'next/headers';
import { getAuthUser } from '@/src/lib/auth';

export async function GET(
  req: NextRequest,
  context: { params: Promise<{ id: string }> } // 👈 Updated to Promise
) {
  try {
    const user = await getAuthUser(cookies());

    if (!user) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const { id } = await context.params; // 👈 Await the params
    const db = await openDb();

    const overallFeedback = await db.get(
      `SELECT overall_feedback, overall_score FROM feedbacks WHERE interview_id = ?`,
      id
    );

    const questionFeedbacks = await db.all(
      `SELECT question, feedback, score FROM question_feedbacks WHERE interview_id = ? ORDER BY id ASC`,
      id
    );

    if (!overallFeedback && questionFeedbacks.length === 0) {
      return NextResponse.json({ message: 'Feedback not found' }, { status: 404 });
    }

    return NextResponse.json({
      overallFeedback: overallFeedback?.overall_feedback ?? null,
      overallScore: overallFeedback?.overall_score ?? null,
      feedbackResults: questionFeedbacks.map((qf: any) => ({
        question: qf.question,
        feedback: qf.feedback,
        score: qf.score,
      })),
    });
  } catch (error) {
    console.error('Error fetching feedback details:', error);
    return NextResponse.json({ error: 'Failed to fetch feedback details' }, { status: 500 });
  }
}