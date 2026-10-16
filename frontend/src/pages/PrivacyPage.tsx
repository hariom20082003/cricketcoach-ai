import { Link } from 'react-router-dom'

export default function PrivacyPage() {
  return (
    <main className="min-h-screen bg-cricket-charcoal text-white px-5 py-12">
      <article className="max-w-3xl mx-auto space-y-7">
        <header>
          <Link to="/" className="text-cricket-green hover:underline text-sm">← CricketCoach AI</Link>
          <h1 className="text-3xl font-bold mt-5">Privacy Notice</h1>
          <p className="text-white/45 text-sm mt-2">For the non-commercial college demonstration.</p>
        </header>

        <section className="card p-6 space-y-3 text-sm text-white/65 leading-relaxed">
          <h2 className="text-white font-semibold">What the app collects</h2>
          <p>Registration collects your email address, username, optional name, batting hand, experience level, and password. The backend stores a bcrypt password hash, never the plaintext password. Login issues a signed bearer session token; the browser keeps that token in local storage so a page refresh does not sign you out.</p>
          <p>If you upload a video, the app stores the video and its original filename, size, duration, dimensions, and recording preferences. Analysis stores the shot label and raw class scores, pose landmarks and measurements, technique observations, job status, and timestamps. Profile changes, session notes, drill completions, and progress records are also stored when you use those features.</p>
        </section>

        <section className="card p-6 space-y-3 text-sm text-white/65 leading-relaxed">
          <h2 className="text-white font-semibold">Where data is stored and processed</h2>
          <p>In a local development run, accounts and analysis data use the configured SQLite database and uploaded videos use the local uploads directory. In the hosted configuration described in the deployment guide, account/session and analysis records use Supabase Postgres; uploaded videos and private model files use a private Supabase Storage bucket; and video analysis runs on the Render backend.</p>
          <p>The backend downloads a private video to temporary server storage while processing it, then removes that temporary copy. Video objects are not published through a public bucket or static URL. The API checks that the signed-in account owns a video or result before returning it. Supabase storage credentials stay in Render environment variables and are never sent to the browser.</p>
          <p>If Hugging Face ZeroGPU inference is enabled, the Render backend forwards the video being analyzed to the configured private Hugging Face Space. The Space processes it for shot classification; this app does not intentionally retain a second copy there, but provider-side processing or logs may apply. Hosted inference can be subject to third-party queueing and daily free-tier limits. Do not upload sensitive or confidential footage to this demonstration.</p>
        </section>

        <section className="card p-6 space-y-3 text-sm text-white/65 leading-relaxed">
          <h2 className="text-white font-semibold">How to delete data</h2>
          <p>Use Delete beside a clip in History to permanently remove its uploaded video and the associated saved analysis, prediction, pose measurements, and observations. Sign out to revoke the current server-side session.</p>
          <p>To remove all application data, use “Delete my account and data” in Profile. This deletes your account, private video objects, analyses, notes, and progress. The app has no separate notes deletion screen. Deletion from application storage does not remove copies you kept on your own device.</p>
        </section>

        <p className="text-white/35 text-xs">This is a student demonstration, not a commercial coaching or medical service. Class scores are uncalibrated, and technique observations are automated measurements or rule-based suggestions rather than expert-validated judgments.</p>
      </article>
    </main>
  )
}
