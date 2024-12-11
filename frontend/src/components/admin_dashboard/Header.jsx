import React from 'react'
import { BsFillBellFill, BsFillEnvelopeFill, BsPersonCircle, BsSearch, BsJustify, BsX, BsXLg }
  from 'react-icons/bs'

function Header({ OpenSidebar, openSidebarToggle }) {
  return (
    <header className='flex items-center justify-between p-4 shadow-md bg-primary'>
      <div className='menu-icon'>
        {openSidebarToggle ?
          <BsXLg className='text-xl cursor-pointer hover:text-secondary' onClick={OpenSidebar} />
          : <BsJustify className='text-xl cursor-pointer hover:text-secondary' onClick={OpenSidebar} />
        }
      </div>  
      <input type="text" name="search" placeholder='Type to search...' className='w-1/3 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-secondary' />
      <div className='flex items-center space-x-4'>
        <span className="text-xl cursor-pointer hover:text-secondary">
          <BsFillBellFill />
        </span>
        <span className="flex text-xl cursor-pointer hover:text-secondary">
          <BsPersonCircle /> username
        </span>
      </div>
    </header>
  )
}

export default Header
