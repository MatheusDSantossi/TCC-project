import React, { useState, useEffect, useRef } from "react";
import { BsChatFill, BsGrid1X2Fill, BsFillArchiveFill } from "react-icons/bs";

import logo from "../../assets/logo.svg";

import "../../App.css";

import * as motion from "motion/react-client";

function Sidebar({ onSidebarToggle, changeComponent }) {
  const [isOpen, setIsOpen] = useState(false);

  const containerRef = useRef(null);

  const { height } = 100;

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
    <div className="bg-primary">
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
          <motion.div style={background} variants={sidebarVariants} />
          <Navigation changeComponent={changeComponent} />
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

const listItemsNav = {
  Dashboard: {
    icon: <BsGrid1X2Fill />,
    component: "dashboard",
  }, // Add your desired icons here
  Product: {
    icon: <BsFillArchiveFill />,
    component: "productRegistration",
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

const Navigation = ({ changeComponent }) => (
  <motion.ul style={list} variants={navVariants}>
    {Object.entries(listItemsNav).map(([label, { icon, component }], i) => (
      <MenuItem
        key={i}
        label={label}
        icon={icon}
        onClick={(e) => {
          e.preventDefault();
          console.log("Change component: ", changeComponent);
          changeComponent(component);
        }}
      />
    ))}
  </motion.ul>
);

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
};

const colors = ["#FF008C", "#D309E1", "#9C1AFF", "#7700FF", "#4400FF"];

const MenuItem = ({ label, icon, onClick }) => {
  // const border = `2px solid ${colors[i]}`;
  return (
    <motion.li
      style={listItem}
      variants={itemVariants}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
    >
      <div style={iconPlaceholder}>{icon}</div>
      <div style={textPlaceholder}>
        <span style={{ color: "white", fontWeight: "bold" }}>{label}</span>
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

const container = {
  position: "relative",
  display: "flex",
  // justifyContent: "flex-start",
  // alignItems: "stretch",
  flex: 1,
  width: 210,
  maxWidth: "100%",
  minHeight: "100vh",
  height: "auto",
  // backgroundColor: "var(--accent)",
  backgroundColor: "#012921",
  borderRadius: 20,
  overflow: "hidden",
  flexDirection: "column",
};

const background = {
  // backgroundColor: "#013D31",
  backgroundColor: "#012921",
  position: "absolute",
  top: 0,
  left: -13,
  bottom: 0,
  width: 300,
};

const toggleContainer = {
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

const list = {
  listStyle: "none",
  padding: 25,
  margin: 0,
  position: "absolute",
  top: 80,
  width: 230,
};

const listItem = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 0,
  margin: 0,
  listStyle: "none",
  marginBottom: 20,
  cursor: "pointer",
};

const iconPlaceholder = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  width: 40,
  height: 40,
  borderRadius: "50%",
  flex: "10x 0",
  marginRight: 10,
};

const textPlaceholder = {
  borderRadius: 5,
  width: 200,
  height: 20,
  flex: 1,
};

export default React.memo(Sidebar);
