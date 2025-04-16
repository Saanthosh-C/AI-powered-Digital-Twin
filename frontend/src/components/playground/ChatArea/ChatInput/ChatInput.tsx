'use client'
import { useState } from 'react'
import { toast } from 'sonner'
import { TextArea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { usePlaygroundStore } from '@/store'
import useAIChatStreamHandler from '@/hooks/useAIStreamHandler'
import { useQueryState } from 'nuqs'
import Icon from '@/components/ui/icon'
import { UploadButton } from '@/utils/uploadthing'
import axios from "axios"

const ChatInput = () => {
  const { chatInputRef } = usePlaygroundStore()

  const { handleStreamResponse } = useAIChatStreamHandler()
  const [selectedAgent] = useQueryState('agent')
  const [inputMessage, setInputMessage] = useState('')
  const isStreaming = usePlaygroundStore((state) => state.isStreaming)
  const handleSubmit = async () => {
    if (!inputMessage.trim()) return

    const currentMessage = inputMessage
    setInputMessage('')

    try {
      await handleStreamResponse(currentMessage)
    } catch (error) {
      toast.error(
        `Error in handleSubmit: ${
          error instanceof Error ? error.message : String(error)
        }`
      )
    }
  }

  return (
    <div className="relative mx-auto mb-1 flex w-full  flex-row  gap-2 max-w-2xl  justify-center gap-x-2 font-geist">
      <TextArea
        placeholder={'Ask anything'}
        value={inputMessage}
        onChange={(e) => setInputMessage(e.target.value)}
        onKeyDown={(e) => {
          if (
            e.key === 'Enter' &&
            !e.nativeEvent.isComposing &&
            !e.shiftKey &&
            !isStreaming
          ) {
            e.preventDefault()
            handleSubmit()
          }
        }}
        className="w-full border border-accent bg-primaryAccent px-4 text-sm text-primary focus:border-accent"
        disabled={!selectedAgent}
        ref={chatInputRef}
      />
      
      <Button
        onClick={handleSubmit}
        disabled={!selectedAgent || !inputMessage.trim() || isStreaming}
        size="icon"
        className="rounded-xl bg-primary p-5 text-primaryAccent"
      >
        <Icon type="send" color="primaryAccent" />
      </Button>
      <UploadButton
        endpoint={"imageUploader"}
        onClientUploadComplete={async(file) => {
          console.log("Files",file)

          const currentFile = file[0].ufsUrl;
          try {
            const response = await axios.post("http://localhost:7777/file",{
              file_url : currentFile
            },{
              headers : {
                "Content-Type": "application/json",

              }
            })

            console.log(response.data)
          } catch (error) {
            console.error(error)
          }

        }}
        onUploadError={(error: Error) => {
          // Do something with the error.
          alert(`ERROR! ${error.message}`);
        }}
      />
    </div>
  )
}

export default ChatInput
