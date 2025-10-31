import React from "react";
import styles from "./Tabs.module.css";

type TabsProps = {
  tabs: {
    label: String;
    index: number;
    Component: React.FC<{ index: number }>;
  }[];
  selectedTab: number;
  onClick: (index: number) => void;
  orientation?: "horizontal" | "vertical";
  className?: string;
};

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
  onClick,
  orientation = "horizontal",
}) => {
  const Panel = tabs && tabs.find((tab) => tab.index === selectedTab);

  console.log("selected tab (tabs): ", selectedTab);
  console.log("tab index (tabs): ", tabs[0].index);

  return (
    <div
      className={
        orientation === "vertical" ? className + " vertical" : className
      }
    >
      <div role="tablist" aria-orientation={orientation}>
        {tabs.map((tab) => {
          return (
            <button
              className={selectedTab === tab.index ? "active" : ""}
              onClick={() => onClick(tab.index)}
              key={tab.index}
              type="button"
              role="tab"
              aria-selected={selectedTab === tab.index}
              aria-controls={`tabpanel-${tab.index}`}
              tabIndex={selectedTab === tab.index ? 0 : -1}
              id={`btn-${tab.index}`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      <div
        role="tabpanel"
        aria-labelledby={`btn-${selectedTab}`}
        id={`tabpanel-${selectedTab}`}
      >
        {Panel && <Panel.Component index={selectedTab} />}
      </div>
    </div>
  );
};

export default Tabs;
