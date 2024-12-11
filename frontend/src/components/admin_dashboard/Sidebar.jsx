import React from 'react'
import {
BsChatFill, BsGrid1X2Fill, BsFillArchiveFill, BsFillGrid3X3GapFill, BsPeopleFill,
BsListCheck, BsMenuButtonWideFill, BsFillGearFill
}
    from 'react-icons/bs'

import logo from '../../assets/logo.svg';    
function Sidebar({ openSidebarToggle, OpenSidebar, changeComponent }) {
    // supermercado
    // products
    // customers
    //inventory
    // reports
    // 

    
    return (
        <aside className={`w-64 bg-primary text-white h-full flex-col p-6 ${openSidebarToggle ? "sidebar-responsive" : "hidden"}`}>
            <div className='mb-6'>
                <div className='sidebar-brand'>
                </div>
                <img src={logo} alt="" />
                {/* <h2 className='text-2xl font-bold' >Hecate</h2> */}
            </div>

            <ul className='space-y-4'>
                <li className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <a onClick={(e) => {
                        changeComponent("dashboard")
                        e.preventDefault();
                    }}>
                        <BsGrid1X2Fill className='icon' /> Dashboard
                    </a>
                </li>
                <li className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <a onClick={(e) => {
                        changeComponent("productRegistration")
                        e.preventDefault();
                        
                    }}>
                        <BsFillArchiveFill className='icon' /> Products
                    </a>
                </li>
                <li className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <a onClick={(e) => {
                        changeComponent("chatbot")
                        e.preventDefault();
                    }}>
                        <BsChatFill className='icon' /> Chatbot
                    </a>
                </li>
                <li className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <a href="">
                        <BsPeopleFill className='icon' /> Customers
                    </a>
                </li>
                <li className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <a href="">
                        <BsListCheck className='icon' /> Inventory
                    </a>
                </li>
                <li className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <a href="">
                        <BsMenuButtonWideFill className='icon' /> Reports
                    </a>
                </li>
                <li className='p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded'>
                    <a href="">
                        <BsFillGearFill className='icon' /> Setting
                    </a>
                </li>
            </ul>
        </aside>
    )
}

export default Sidebar