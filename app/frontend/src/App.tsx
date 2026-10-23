import { Navigate, Route, Routes } from "react-router-dom";
import ProductLayout from "./layouts/ProductLayout";
import DevLayout from "./layouts/DevLayout";
import Landing from "./pages/product/Landing";
import HowItWorks from "./pages/product/HowItWorks";
import ProductAbout from "./pages/product/About";
import ReviewPage from "./pages/Review";
import DevOverview from "./pages/dev/Overview";
import ResultsPage from "./pages/Results";
import ExperimentsPage from "./pages/Experiments";
import DatasetPage from "./pages/Dataset";
import TestingPage from "./pages/Testing";
import RelatedWorkPage from "./pages/RelatedWork";

const NotFound = () => <div className="p-10 text-center text-muted">Page not found.</div>;

export default function App() {
  return (
    <Routes>
      <Route element={<ProductLayout />}>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<ReviewPage />} />
        <Route path="/how-it-works" element={<HowItWorks />} />
        <Route path="/about" element={<ProductAbout />} />
      </Route>
      <Route path="/dev" element={<DevLayout />}>
        <Route index element={<DevOverview />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="experiments" element={<ExperimentsPage />} />
        <Route path="dataset" element={<DatasetPage />} />
        <Route path="testing" element={<TestingPage />} />
        <Route path="related-work" element={<RelatedWorkPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
      <Route path="/review" element={<Navigate to="/app" replace />} />
      {["results", "experiments", "dataset", "testing", "related-work"].map((p) => <Route key={p} path={`/${p}`} element={<Navigate to={`/dev/${p}`} replace />} />)}
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
