import React from 'react'

function MenuRegistration() {
  return (
    <div>
      <div className="flex">
        <h1>Registration Menu</h1>
        <form action="">
          <label htmlFor="name">Name:</label>
          <input type="text" id="name" name="name" required />
          <label htmlFor="name">Description:</label>
          <input type="text" id="description" name="description" required />
        </form>
      </div>
    </div>
  )
}

export default MenuRegistration
