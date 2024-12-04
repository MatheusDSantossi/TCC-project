import React from 'react'
import {
BsCart3, BsGrid1X2Fill, BsFillArchiveFill, BsFillGrid3X3GapFill, BsPeopleFill,
BsListCheck, BsMenuButtonWideFill, BsFillGearFill
}
    from 'react-icons/bs'
import { useNavigate } from 'react-router-dom'


function Sidebar({ openSidebarToggle, OpenSidebar }) {
    const navigate = useNavigate();
    
    const handleClick = () => {
        navigate('/productregistration')
    }
    
    return (
        <aside className={`w-64 bg-primary text-white h-full flex-col p-6 ${openSidebarToggle ? "sidebar-responsive" : "sidebar"}`}>
            <div className='mb-6'>
                <div className='sidebar-brand'>
                </div>
                <h2 className='text-2xl font-bold' >Hecate</h2>
            </div>

            <ul className='space-y-4'>
                <li className='cursor-pointer hover:text-secondary hover:bg-primary p-2'>
                    <a href="">
                        <BsGrid1X2Fill className='icon' /> Dashboard
                    </a>
                </li>
                <li className='cursor-pointer hover:text-secondary hover:bg-primary p-2'>
                    <a href="/productregistration">
                        <BsFillArchiveFill className='icon' /> Products
                    </a>
                </li>
                <li className='cursor-pointer hover:text-secondary hover:bg-primary p-2'>
                    <a href="">
                        <BsFillGrid3X3GapFill className='icon' /> Categories
                    </a>
                </li>
                <li className='cursor-pointer hover:text-secondary hover:bg-primary p-2'>
                    <a href="">
                        <BsPeopleFill className='icon' /> Customers
                    </a>
                </li>
                <li className='cursor-pointer hover:text-secondary hover:bg-primary p-2'>
                    <a href="">
                        <BsListCheck className='icon' /> Inventory
                    </a>
                </li>
                <li className='cursor-pointer hover:text-secondary hover:bg-primary p-2'>
                    <a href="">
                        <BsMenuButtonWideFill className='icon' /> Reports
                    </a>
                </li>
                <li className='cursor-pointer hover:text-secondary hover:bg-primary p-2'>
                    <a href="">
                        <BsFillGearFill className='icon' /> Setting
                    </a>
                </li>
            </ul>
        </aside>
    )
}

export default Sidebar