import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { BrowserRouter as Router } from "react-router-dom";
import "./index.css";
import { ExceptionProvider } from "./context/ExceptionContext.js";
import { ErrorBoundary } from "./exceptions/ErrorBoundary.js";

createRoot(document.getElementById("root")).render(
  <ExceptionProvider>
    <ErrorBoundary>
      <Router>
        <StrictMode>
          <App />
        </StrictMode>
      </Router>
    </ErrorBoundary>
  </ExceptionProvider>
);
