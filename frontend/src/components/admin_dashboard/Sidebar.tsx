import React, { useState, useRef, type CSSProperties, type ReactNode } from "react";
import {
  BsChatFill,
  BsTools,
  BsFillArchiveFill,
} from "react-icons/bs";

import logo from "../../assets/logo.svg";

import * as motion from "motion/react-client";
import type { NavTree } from "../../types/NavTypes";
import { IoIosArrowForward } from "react-icons/io";

function Sidebar({ onSidebarToggle, changeComponent }) {
  // States
  const [isOpen, setIsOpen] = useState(false);

  const containerRef = useRef(null);

  const height = 100;

  const sidebarVariants = {
    open: (height = 1000) => ({
      clipPath: `circle(${height * 2 + 200}px at 40px 40px)`,
      transition: {
        type: "spring",
        stiffness: 20,
        restDelta: 2,
      },
    }),
    closed: {
      clipPath: "circle(30px at 40px 40px)",
      transition: {
        delay: 0.2,
        type: "spring",
        stiffness: 400,
        damping: 40,
      },
    },
  };
  return (
    <div className="h-screen bg-primary shadow-3xl">
      <div style={container}>
        <motion.nav
          initial={false}
          animate={isOpen ? "open" : "closed"}
          custom={height}
          ref={containerRef}
        >
          {/* Logo Section */}
          <div className="flex justify-center gap-4 p-4 mb-2 bg-primary">
            <img src={logo} alt="brand" className="w-28 h-28" />
          </div>
          <motion.div
            style={background}
            variants={sidebarVariants}
            className={!isOpen ? "pointer-events-none" : ""}
          />
          <div style={{ pointerEvents: isOpen ? "auto" : "none" }}>
            <Navigation changeComponent={changeComponent} />
          </div>
          <MenuToggle toggle={() => setIsOpen(!isOpen)} />
        </motion.nav>
      </div>
    </div>
  );
}

const navVariants = {
  open: {
    transition: { staggerChildren: 0.07, delayChildren: 0.2 },
  },
  closed: {
    transition: { staggerChildren: 0.05, staggerDirection: -1 },
  },
};

const listItemsNav: NavTree = {
  // Dashboard: {
  //   icon: <BsGrid1X2Fill />,
  //   component: "dashboard",
  // },
  Produtos: {
    icon: <BsFillArchiveFill />,
    subitems: {
      Cadastro: {
        component: "productRegistration",
      },
      Lista: {
        component: "productList",
      },
    },
  },
  Serviços: {
    icon: <BsTools />,
    subitems: {
      Cadastro: {
        component: "companyInfoRegistration",
      },
      Lista: {
        component: "businessDetailList",
      },
    },
  },
  Chatbot: {
    icon: <BsChatFill />,
    component: "chatbot",
  },
  // Settings: {
  //   icon: <MdLogout />,
  //   handler: (e, handleLogout) => {
  //     handleLogout;
  //     e.preventDefault();
  //   }
  // },
};

const Navigation = ({ changeComponent }) => {
  const [openSubmenu, setOpenSubmenu] = useState(null);

  return (
    <motion.ul style={list} variants={navVariants}>
      {Object.entries(listItemsNav).map(([label, item], i) => (
        <div key={i}>
          <MenuItem
            label={label}
            icon={item.icon}
            hasSubitems={!!item.subitems}
            isOpen={openSubmenu === label}
            onClick={(e) => {
              e.preventDefault();
              console.log("Change component: ", changeComponent);
              if (item.subitems) {
                setOpenSubmenu(openSubmenu === label ? null : label);
              } else {
                changeComponent(item.component);
              }
            }}
          />
          {item.subitems && openSubmenu === label && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              {Object.entries(item.subitems).map(([subLabel, subItem], j) => (
                <SubMenuItem
                  key={j}
                  label={subLabel}
                  onClick={(e) => {
                    e.preventDefault();
                    changeComponent(subItem.component);
                  }}
                />
              ))}
            </motion.div>
          )}
        </div>
      ))}
    </motion.ul>
  );
};

const itemVariants = {
  open: {
    y: 0,
    opacity: 1,
    transition: {
      y: { stiffness: 1000, velocity: -100 },
    },
  },
  closed: {
    y: 50,
    opacity: 0,
    transition: {
      y: { stiffness: 1000 },
    },
  },
  submenu: {
    y: 0,
    opacity: 1,
    transition: {
      delay: 0.1,
      duration: 0.2,
    },
  },
};

// const colors = ["#FF008C", "#D309E1", "#9C1AFF", "#7700FF", "#4400FF"];

const MenuItem = ({ label, icon, onClick, hasSubitems, isOpen }) => {
  // const border = `2px solid ${colors[i]}`;
  return (
    <motion.li
      style={listItem}
      variants={itemVariants}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`hover:bg-tertiary ${isOpen ? "bg-tertiary" : ""}`}
    >
      <div style={iconPlaceholder}>{icon}</div>
      <div style={{ ...textPlaceholder, position: "relative" }}>
        <span className="mr-2" style={{ color: "white", fontWeight: "bold" }}>{label}</span>
        {hasSubitems && (
          <motion.span
          className="ml-2 mt-[2px]"
            style={{
              position: "absolute",
              top: 3,
              right: 0,
              color: "white",
              transform: isOpen ? "rotate(90deg)" : "rotate(0deg)",
            }}
            initial={{ rotate: 0 }}
            animate={{ rotate: isOpen ? 90 : 0 }}
          >
            <IoIosArrowForward />
          </motion.span>
        )}
      </div>
    </motion.li>
  );
};

const SubMenuItem = ({ label, onClick }) => {
  return (
    <motion.li
      style={{ ...listItem, marginLeft: "30px", padding: "8px 25px 8px 45px" }}
      variants={itemVariants}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.99 }}
      onClick={onClick}
    >
      <div className="flex items-stretch group" style={textPlaceholder}>
        <div className="w-1 h-6 mr-2 transition-opacity rounded opacity-0 bg-secondary group-hover:opacity-100" />
        <span style={{ color: "white", fontWeight: "normal" }}>{label}</span>
      </div>
    </motion.li>
  );
};

const Path = (props) => (
  <motion.path
    // fill="transparent"
    strokeWidth="3"
    // stroke="hsl(0, 0%, 18%)"
    strokeLinecap="round"
    {...props}
  />
);

const MenuToggle = ({ toggle }) => (
  <button style={toggleContainer} onClick={toggle}>
    <svg className="burger-icon" width="23" height="23" viewBox="0 0 23 23">
      <Path
        variants={{
          closed: { d: "M 2 2.5 L 20 2.5" },
          open: { d: "M 3 16.5 L 17 2.5" },
        }}
      />
      <Path
        d="M 2 9.423 L 20 9.423"
        variants={{
          closed: { opacity: 1 },
          open: { opacity: 0 },
        }}
        transition={{ duration: 0.1 }}
      />
      <Path
        variants={{
          closed: { d: "M 2 16.346 L 20 16.346" },
          open: { d: "M 3 2.5 L 17 16.346" },
        }}
      />
    </svg>
  </button>
);

/**
 * ==============   Styles   ================
 */

const container: CSSProperties = {
  position: "relative",
  display: "flex",
  height: "100vh",
  // justifyContent: "flex-start",
  // alignItems: "stretch",
  flex: 1,
  width: 210,
  maxWidth: "100%",
  // minHeight: "109vh",
  // height: "auto",
  // backgroundColor: "var(--accent)",
  backgroundColor: "#012921",
  overflow: "hidden",
  flexDirection: "column",
};

const background: CSSProperties = {
  // backgroundColor: "#013D31",
  backgroundColor: "#012921",
  position: "absolute",
  top: 0,
  left: -13,
  bottom: 0,
  width: 300,
};

const toggleContainer: CSSProperties = {
  outline: "none",
  border: "none",
  WebkitUserSelect: "none",
  MozUserSelect: "none",
  cursor: "pointer",
  position: "absolute",
  top: 18,
  left: 15,
  width: 50,
  height: 50,
  borderRadius: "50%",
  background: "transparent",
  fill: "yellow",
};

const list: CSSProperties = {
  listStyle: "none",
  padding: "25px 0",
  margin: 0,
  position: "relative",
  top: -80,
  width: "100%",
};

const listItem: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "flex-start",
  padding: "8px 25px",
  margin: 0,
  listStyle: "none",
  marginBottom: 8,
  cursor: "pointer",
  width: "100%",
};

const iconPlaceholder: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  width: 40,
  height: 40,
  borderRadius: "50%",
  flex: "10x 0",
  marginRight: 10,
  color: "white",
};

const textPlaceholder: CSSProperties = {
  borderRadius: 5,
  width: 200,
  height: "auto",
  flex: 1,
};

export default React.memo(Sidebar);
