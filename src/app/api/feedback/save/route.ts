import { NextResponse } from 'next/server';
import { openDb } from '@/src/lib/db';
import { getAuthUser } from '@/src/lib/auth';
import { cookies } from 'next/headers';

export async function POST(request: Request) {
  try {
    const user = await getAuthUser(cookies());

    if (!user) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const { interviewId, feedbackResults, jobTitle } = await request.json();

    if (!interviewId || !feedbackResults) {
      return NextResponse.json({ error: 'Interview ID and feedback results are required' }, { status: 400 });
    }

    const db = await openDb();

    // Calculate overall score (e.g., average of all scores)
    const totalScore = feedbackResults.reduce((sum: number, item: any) => sum + item.score, 0);
    const overallScore = feedbackResults.length > 0 ? Math.round(totalScore / feedbackResults.length) : 0;

    // Summarize feedback (e.g., concatenate all feedback strings)
    const summary = feedbackResults.map((item: any) => item.feedback).join('\n\n');

    // Save overall feedback
    const result = await db.run(
      `INSERT INTO feedbacks (interview_id, overall_score, summary)
       VALUES (?, ?, ?)`, 
      interviewId,
      overallScore,
      summary
    );

    // Update the interview status to 'Completed' and save job title
    await db.run(
      `UPDATE interviews SET status = 'Completed', job_title = ? WHERE id = ?`,
      jobTitle,
      interviewId
    );

    // Save individual question feedbacks
    for (const item of feedbackResults) {
      await db.run(
        `INSERT INTO question_feedbacks (feedback_id, question, user_answer, feedback)
         VALUES (?, ?, ?, ?)`, 
        result.lastID,
        item.question, // Assuming feedbackResults now includes the original question
        item.user_answer, // Assuming feedbackResults now includes the user's answer
        item.feedback
      );
    }

    return NextResponse.json({ feedbackId: result.lastID });
  } catch (error) {
    console.error('Error saving feedback:', error);
    return NextResponse.json({ error: 'Failed to save feedback' }, { status: 500 });
  }
}