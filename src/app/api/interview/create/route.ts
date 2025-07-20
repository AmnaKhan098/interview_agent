import { NextResponse } from 'next/server';
// import { openDb } from '@/lib/db';
import { getAuthUser } from '@/src/lib/auth';
import { cookies } from 'next/headers';
import { openDb } from '@/src/lib/db';

export async function POST(request: Request) {
  try {
    const user = await getAuthUser(cookies());

    if (!user) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const userId = user.id;


    const { jobTitle, jobDescription, techExpertise, yearsOfExperience } = await request.json();

    if (!jobTitle) {
      return NextResponse.json({ error: 'Job title is required' }, { status: 400 });
    }

    const db = await openDb();
    const result = await db.run(
      `INSERT INTO interviews (user_id, job_title, job_description, tech_expertise, years_experience, status)
       VALUES (?, ?, ?, ?, ?, ?)`, 
      userId,
      jobTitle,
      jobDescription,
      techExpertise,
      yearsOfExperience,
      'In Progress'
    );

    return NextResponse.json({ interviewId: result.lastID });
  } catch (error) {
    console.error('Error creating interview:', error);
    return NextResponse.json({ error: 'Failed to create interview' }, { status: 500 });
  }
}