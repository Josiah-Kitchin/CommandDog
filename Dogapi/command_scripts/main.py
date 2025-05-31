
import asyncio 

from command import DogController


async def main():
    c = DogController()
    c.start()
    c.set_mode("ai")

    while True: 
        user_input = input("Input: ")
        if "handstand" in user_input: 
            c.handstand(on=True)


if __name__ == "__main__": 
    asyncio.run(main())



    




