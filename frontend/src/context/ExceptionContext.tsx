import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import {
  ExceptionVariant,
  type ExceptionPayload,
} from "../exceptions/ExceptionVariantsTypes";
import ExceptionMainComponent from "../exceptions/ExceptionMainComponent";

type ExceptionContextType = {
  current?: ExceptionPayload | null;
  showException: (payload: ExceptionPayload) => void;
  hideException: () => void;
};

const ExceptionContext = createContext<ExceptionContextType | undefined>(
  undefined
);

export const ExceptionProvider: React.FC<React.PropsWithChildren<{}>> = ({
  children,
}) => {
  const [current, setCurrent] = useState<ExceptionPayload | null>(null);

  const showException = useCallback((payload: ExceptionPayload) => {
    // default variant to ERROR if not provided
    setCurrent({
      ...payload,
      variant: payload.variant ?? ExceptionVariant.ERROR,
    });
  }, []);

  const hideException = useCallback(() => {
    setCurrent(null);
  }, []);

  //   Global handlers: unhandledrejection and window.onerror
  useEffect(() => {
    const onUnhandledRejection = (ev: PromiseRejectionEvent) => {
      const reason = ev.reason;
      // show compact error message
      showException({
        text: reason?.message ?? "Unhandled promise rejection",
        error: reason,
        variant: ExceptionVariant.ERROR,
      });
    };

    const onWindowError = (
      message: any,
      source?: string,
      lineno?: number,
      colno?: number,
      error?: Error
    ) => {
      showException({
        text: (error?.message ?? message ?? "Unexpected error").toString(),
        error: error ?? { message, source, lineno, colno },
        variant: ExceptionVariant.ERROR,
      });
      // returning false allows the error to propagate to console as usual
      return false;
    };

    window.addEventListener("unhandledrejection", onUnhandledRejection);
    // @ts-ignore - older signatures for onerror - still safe
    window.onerror = onWindowError;

    return () => {
      window.removeEventListener("unhandledrejection", onUnhandledRejection);
      // restore default
      // @ts-ignore
      window.onerror = null;
    };
  }, [showException]);

  return (
    <ExceptionContext.Provider
      value={{ current, showException, hideException }}
    >
      {children}
      {/* Render modal at top level so any page can trigger it */}
      {current && (
        <ExceptionMainComponent
          text={current.text}
          error={current.error ?? { message: current.text }}
          variant={current.variant as ExceptionVariant}
          onClose={hideException}
        />
      )}
    </ExceptionContext.Provider>
  );
};

export function useException() {
  const ctx = useContext(ExceptionContext);
  if (!ctx)
    throw new Error("useException must be used within ExceptionProvider");
  return ctx;
}
