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

  const handleOpen = (value) => {
    setOpen(open === value ? 0 : value);
  };
  
  const toggleDrawer = () => {
    console.log("IsDrawerOpen: ", isDrawerOpen);
    setIsDrawerOpen(!isDrawerOpen);
    onSidebarToggle(!isDrawerOpen);
  };

  // const openDrawer = () => {
  //   setIsDrawerOpen(true);
  // };
  // const closeDrawer = () => {
  //   setIsDrawerOpen(false);
  // };

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
      <Drawer open={isDrawerOpen} onClose={toggleDrawer}>
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
            <ListItem>
              <ListItemPrefix
                className="flex p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded"
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
            <ListItem>
              <ListItemPrefix
                onClick={(e) => {
                  changeComponent("productRegistration");
                  toggleDrawer();
                  e.preventDefault();
                }}
                className="flex p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded"
              >
                <BsFillArchiveFill className="w-5 h-5 mr-3" />
                Products
              </ListItemPrefix>
            </ListItem>
            <ListItem>
              <ListItemPrefix
                onClick={(e) => {
                  changeComponent("chatbot");
                  toggleDrawer();
                  e.preventDefault();
                }}
                className="flex p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded"
              >
                <BsChatFill className="w-5 h-5 mr-3" />
                Chatbot
              </ListItemPrefix>
            </ListItem>
            <ListItem>
              <ListItemPrefix
                onClick={handleLogout}
                className="flex p-2 cursor-pointer hover:text-primary hover:bg-red-500 hover:rounded"
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
            <Typography variant="h6" className="mb-1">
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
    </div>
  );
}

export default React.memo(Sidebar);
