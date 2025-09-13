import React, { createContext, useCallback, useState } from "react";
import {
  ExceptionVariant,
  type ExceptionPayload,
} from "../exceptions/ExceptionVariantsTypes";

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

  //   Global handlers: unganlerejection and window.onerror
};
