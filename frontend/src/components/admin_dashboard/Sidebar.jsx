import React, { useState } from "react";
import {
  BsChatFill,
  BsGrid1X2Fill,
  BsFillArchiveFill,
  BsJustify,
  BsXLg,
  BsBookmark,
} from "react-icons/bs";

import logo from "../../assets/logo.svg";
import { MdLogout } from "react-icons/md";

import AxiosInstance from "../AxiosInstance";
import { useNavigate } from "react-router-dom";

import { AnimatePresence, motion } from "framer-motion";

import {
  Alert,
  Card,
  Chip,
  Drawer,
  IconButton,
  Input,
  List,
  ListItem,
  ListItemPrefix,
  ListItemSuffix,
  Typography,
} from "@material-tailwind/react";

function Sidebar({ onSidebarToggle, changeComponent }) {
  const navigate = useNavigate();

  const [open, setOpen] = useState(0);
  const [openAlert, setOpenAlert] = useState(true);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const handleLogout = () => {
    AxiosInstance.post(`logoutall/`, {}).then(() => {
      // Clear user session data and navigate to login pagel
      localStorage.removeItem("Token");

      // Navigate to login page
      navigate("/login");
    });
  };

  const toggleDrawer = () => {
    console.log("IsDrawerOpen: ", isDrawerOpen);
    setIsDrawerOpen(!isDrawerOpen);
    onSidebarToggle(!isDrawerOpen);
  };

  const sidebarVariants = {
    hidden: {
      x: "-100%",
      opacity: 0,
    },
    visible: {
      x: "0%",
      opacity: 1,
      transition: { duration: 0.3, ease: "easeOut" },
    },
    exit: {
      x: "-100%",
      opacity: 0,
      transition: { duration: 0.3, ease: "easeIn" },
    },
  };

  return (
    <div className="absolute z-10 bg-primary">
      <IconButton
        variant="text"
        size="lg"
        onClick={toggleDrawer}
        className="text-xl cursor-pointer hover:text-secondary"
      >
        {isDrawerOpen ? <BsXLg /> : <BsJustify />}
      </IconButton>
      <AnimatePresence>
        {isDrawerOpen && (
          <Drawer
            key="box"
            open={isDrawerOpen}
            onClose={toggleDrawer}
            variants={sidebarVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="fixed top-0 left-0 z-20 w-64 h-screen shadow-md bg-primary"
          >
            {/* Sidebar Content */}
            <Card
              color="transparent"
              shadow={false}
              className="h-[calc(100vh-2rem)] w-full p-4 bg-primary"
            >
              <div className="flex justify-center gap-4 p-4 mb-2 bg-primary">
                <img src={logo} alt="brand" className="size-28" />
              </div>
              <hr className="my-2 border-blue-gray-50" />
              <List>
                <ListItem className="flex p-2 mt-4 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded">
                  <ListItemPrefix className="flex"
                    onClick={(e) => {
                      changeComponent("dashboard");
                      toggleDrawer();
                      e.preventDefault();
                    }}
                  >
                    <BsGrid1X2Fill className="w-5 h-5 mr-3" />
                    Dashaboard
                  </ListItemPrefix>
                  <ListItemSuffix>
                    <Chip
                      value="14"
                      size="sm"
                      variant="ghost"
                      color="blue-gray"
                      className="rounded-full"
                    />
                  </ListItemSuffix>
                </ListItem>
                <ListItem className="flex p-2 mt-4 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded">
                  <ListItemPrefix
                    onClick={(e) => {
                      changeComponent("productRegistration");
                      toggleDrawer();
                      e.preventDefault();
                    }}
                    className="flex"
                  >
                    <BsFillArchiveFill className="w-5 h-5 mr-3" />
                    Products
                  </ListItemPrefix>
                </ListItem>
                <ListItem className="flex p-2 mt-4 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded">
                  <ListItemPrefix
                    onClick={(e) => {
                      changeComponent("chatbot");
                      toggleDrawer();
                      e.preventDefault();
                    }}
                    className="flex"
                  >
                    <BsChatFill className="w-5 h-5 mr-3" />
                    Chatbot
                  </ListItemPrefix>
                </ListItem>
                <ListItem className="flex p-2 mt-4 cursor-pointer hover:text-primary hover:bg-red-500 hover:rounded">
                  <ListItemPrefix
                    onClick={handleLogout}
                    className="flex"
                  >
                    <MdLogout className="w-5 h-5 mr-3" />
                    Log Out
                  </ListItemPrefix>
                </ListItem>
              </List>
              <Alert
                open={openAlert}
                className="p-2 mt-auto bg-secondary text-primary"
                onClose={() => setOpenAlert(false)}
              >
                <BsBookmark className="w-12 h-12 mb-4" />
                <Typography variant="h4" className="mb-1">
                  News
                </Typography>
                <Typography variant="small" className="font-normal opacity-80">
                  Check the products to new edits.
                </Typography>
                <div className="flex gap-3 mt-4">
                  <Typography
                    as="a"
                    href="#"
                    variant="small"
                    className="font-medium opacity-80"
                    onClick={() => setOpenAlert(false)}
                  >
                    Dismiss
                  </Typography>
                  <Typography
                    as="a"
                    href="#"
                    variant="small"
                    className="font-medium"
                  >
                    Know more
                  </Typography>
                </div>
              </Alert>
            </Card>
          </Drawer>
        )}
      </AnimatePresence>
    </div>
  );
}

export default React.memo(Sidebar);
