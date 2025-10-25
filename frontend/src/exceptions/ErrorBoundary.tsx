import React, { type ReactNode } from "react";
import { useException } from "../context/ExceptionContext";
import { ExceptionVariant } from "./ExceptionVariantsTypes";

class ErrorBoundaryInner extends React.Component<
  { children?: ReactNode },
  { hasError: boolean }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(_: Error) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // noop - will be handled by wrapper function that has hook access, we'll keep here in case we want logging to external service
    // console.error("ErrorBoundary caught: ", error, info)
  }

  render() {
    if (this.state.hasError) {
      // render nothing; provider will show modal via wrapper
      return null;
    }
    return this.props.children ?? null;
  }
}

// wrapper functional component to call useException() hook and pass errors
export const ErrorBoundary: React.FC<React.PropsWithChildren<{}>> = ({
  children,
}) => {
  const { showException } = useException();

  // a ref callback to attach componentDidCatch behavior
  // simpler: use a componentDidCatch shim via the class above and useEffect to detect error via window.onerror
  // but we can also wrap render inside a try/catch:
  try {
    return <ErrorBoundaryInner>{children}</ErrorBoundaryInner>;
  } catch (err: any) {
    showException({
      text: err?.message ?? "An unexpected error occurred",
      error: err,
      variant: ExceptionVariant.ERROR,
    });
    return null;
  }
};
