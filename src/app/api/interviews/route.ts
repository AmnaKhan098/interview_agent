import { NextResponse } from 'next/server';
interface Interview {
  id: string;
  jobTitle: string;
  date: string;
  status: string;
}

export async function GET() {
  // Mock interview data
  const interviews: Interview[] = [
    {
      id: '1',
      jobTitle: 'Senior Frontend Developer',
      date: '2024-03-15',
      status: 'Completed'
    },
    {
      id: '2',
      jobTitle: 'Full Stack Engineer',
      date: '2024-03-16',
      status: 'Completed'
    },
    {
      id: '3',
      jobTitle: 'React Developer',
      date: '2024-03-17',
      status: 'In Progress'
    }
  ];

  return NextResponse.json(interviews);
}