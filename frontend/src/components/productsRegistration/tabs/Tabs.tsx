import React, { useEffect, useMemo, useState } from "react";
import styles from "./Tabs.module.css";

export type TabItem = {
  label: string;
  index: number | string;
  Component?: React.ComponentType<{ index: number }>;
};

export type TabsProps = {
  tabs: TabItem[];
  selectedTab?: number | string;
  defaultTab?: number | string;
  onClick: (index: number) => void;
  orientation?: "horizontal" | "vertical";
  className?: string;
};

const toNumber = (v?: number | string | null) =>
  typeof v === "number" ? v : v ? Number(v) : NaN;

/**
 * Available Props
 * @param className string
 * @param tab Array of objects
 * @param selectedTab number
 * onClick Function to set the active tab
 * @param orientation Tab orientation Vertical | Horizontal
 * @returns ReactNodeElement
 */

const Tabs: React.FC<TabsProps> = ({
  className = `${styles.tabsComponent}`,
  tabs = [],
  selectedTab = 0,
  defaultTab,
  onClick,
  orientation = "horizontal",
}) => {
  // Coerce tabs to internal representation with numeric indexes
  const normalizedTabs = useMemo(
    () =>
      tabs.map((t) => ({
        ...t,
        _index: toNumber(t.index),
      })),
    [tabs]
  );

  // Internal state only used if `selectedTab` prop is not provided (uncontrolled)
  const [internal, setInternal] = useState<number>(() => {
    if (defaultTab !== undefined) return toNumber(defaultTab);

    // fallback to first tab index (or 0)
    return normalizedTabs.length ? normalizedTabs[0]._index : 0;
  });

  // Active index: prefer controlled prop if present
  const activeIndex = Number(
    selectedTab !== undefined ? toNumber(selectedTab) : internal
  );

  // Ensure we have a sensible default (if parent passed an invalid index)
  useEffect(() => {
    if (Number.isNaN(activeIndex) && normalizedTabs.length) {
      const fallback = normalizedTabs[0]._index;
      if (selectedTab === undefined) setInternal(fallback);
      if (typeof onClick === "function") onClick(fallback);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  });

  const handleClick = (idx: number) => {
    if (selectedTab === undefined) setInternal(idx); // uncontrolled: update internal state
    if (onClick) onClick(idx);
  };

  // Keyboard navigation: left/right arrows (pass)

  const ActivePanel = normalizedTabs.find((t) => t._index === activeIndex);
  return (
    <div
      className={
        orientation === "vertical" ? `${className} vertical` : className
      }
    >
      <div
        role="tablist"
        aria-orientation={orientation}
        className={styles.tabsList}
      >
        {normalizedTabs.map((tab) => {
          const isActive = tab._index === activeIndex;

          return (
            <button
              key={tab.index}
              type="button"
              role="tab"
              aria-selected={isActive}
              aria-controls={`tabpanel-${tab._index}`}
              id={`btn-${tab._index}`}
              tabIndex={isActive ? 0 : -1}
              className={`${
                isActive
                  ? `${styles.tabButton} ${styles.active}`
                  : `${styles.tabButton}`
              }`}
              onClick={() => handleClick(tab._index)}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      <div
        role="tabpanel"
        id={`tabpanel-${activeIndex}`}
        aria-labelledby={`btn-${activeIndex}`}
        className={styles.tabPanel}
      >
        {ActivePanel?.Component ? (
          <ActivePanel.Component index={activeIndex} />
        ) : null}
      </div>
    </div>
  );
};

export default Tabs;
