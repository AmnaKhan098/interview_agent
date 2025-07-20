export default function AboutPage() {
  return (
    <div className="container mx-auto px-4 py-12">
      {/* Header Section */}
      <section className="text-center mb-16">
        <h1 className="text-4xl font-bold mb-6">Our Mission</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          We're on a mission to democratize technical interview preparation by leveraging
          the power of AI to provide personalized, effective practice experiences for
          developers at all levels.
        </p>
      </section>

      {/* Team Section */}
      <section className="py-12">
        <h2 className="text-3xl font-bold text-center mb-12">Meet the Team</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Team Member 1 */}
          <div className="flex flex-col items-center">
            <div className="w-48 h-48 bg-gray-200 rounded-full mb-4"></div>
            <h3 className="text-xl font-semibold mb-1">Sarah Johnson</h3>
            <p className="text-gray-600">Lead Developer</p>
          </div>

          {/* Team Member 2 */}
          <div className="flex flex-col items-center">
            <div className="w-48 h-48 bg-gray-200 rounded-full mb-4"></div>
            <h3 className="text-xl font-semibold mb-1">Michael Chen</h3>
            <p className="text-gray-600">AI Engineer</p>
          </div>

          {/* Team Member 3 */}
          <div className="flex flex-col items-center">
            <div className="w-48 h-48 bg-gray-200 rounded-full mb-4"></div>
            <h3 className="text-xl font-semibold mb-1">Emily Rodriguez</h3>
            <p className="text-gray-600">Product Designer</p>
          </div>
        </div>
      </section>
    </div>
  );
}