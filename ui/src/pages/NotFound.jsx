import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <section className="panel">
      <p className="eyebrow">Lost</p>
      <h2>Page not found</h2>
      <p>That route does not exist yet. Use the navigation to get back.</p>
      <Link className="button" to="/user/create">
        Return to Create
      </Link>
    </section>
  );
}
