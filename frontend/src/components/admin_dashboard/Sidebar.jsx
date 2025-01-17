import React from 'react'
import {
BsChatFill, BsGrid1X2Fill, BsFillArchiveFill, BsPeopleFill,
BsListCheck, BsMenuButtonWideFill, BsFillGearFill
}
    from 'react-icons/bs'

import logo from '../../assets/logo.svg';    
import { MdLogout } from "react-icons/md";

import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';

import AxiosInstance from '../AxiosInstance';
import { useNavigate } from 'react-router-dom';

function Sidebar({ openSidebarToggle, OpenSidebar, changeComponent }) {
    // supermercado
    // products
    // customers
    //inventory
    // reports
    // 

    const navigate = useNavigate();

    const handleLogout = () => {
        AxiosInstance.post(`logoutall/`, {
            
        })
       .then(() => {
         // Clear user session data and navigate to login pagel
         localStorage.removeItem('Token');

         // Navigate to login page
         navigate("/login")
       })
    }
    
    return (
        <aside className={`w-64 bg-primary text-white h-full flex-col p-6 ${openSidebarToggle ? "sidebar-responsive" : "hidden"}`}>
            <div className='mb-6'>
                <div className='sidebar-brand'>
                </div>
                <img src={logo} alt="" />
                {/* <h2 className='text-2xl font-bold' >Hecate</h2> */}
            </div>

            <List className='space-y-4'>
                <ListItem className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <ListItemButton onClick={(e) => {
                        changeComponent("dashboard")
                        e.preventDefault();
                    }}>
                        <BsGrid1X2Fill className='mr-2 icon' /> Dashboard
                    </ListItemButton>
                </ListItem>
                <ListItem className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <ListItemButton onClick={(e) => {
                        changeComponent("productRegistration")
                        e.preventDefault();
                        
                    }}>
                        <BsFillArchiveFill className='mr-2 icon' /> Products
                    </ListItemButton>
                </ListItem>
                <ListItem className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <ListItemButton onClick={(e) => {
                        changeComponent("chatbot")
                        e.preventDefault();
                    }}>
                        <BsChatFill className='mr-2 icon' /> Chatbot
                    </ListItemButton>
                </ListItem>
                <ListItem className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <ListItemButton onClick={(e) => {
                        changeComponent("userList")
                        e.preventDefault();
                    }}>
                        <BsPeopleFill className='mr-2 icon' /> Customers
                    </ListItemButton>
                </ListItem>
                <ListItem className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <ListItemButton href="">
                        <BsListCheck className='mr-2 icon' /> Inventory
                    </ListItemButton>
                </ListItem>
                <ListItem className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <ListItemButton href="">
                        <BsMenuButtonWideFill className='mr-2 icon' /> Reports
                    </ListItemButton>
                </ListItem>
                <ListItem className='p-2 cursor-pointer color-red hover:bg-red-500 hover:rounded'>
                    <ListItemButton onClick={handleLogout}>
                        <MdLogout className='mr-2 icon' /> Logout
                    </ListItemButton>
                </ListItem>
            </List>
        </aside>
    )
}

export default Sidebar